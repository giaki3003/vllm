# SPDX-License-Identifier: Apache-2.0

import dataclasses
import weakref 
import inspect # For execute_model signature check
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Union) 

import torch

from vllm.config import VllmConfig 
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry 
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache 
from vllm.model_executor.layers.sampler import (SamplerOutput, get_sampler)
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs, MultiModalRegistry) 
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata

from vllm.worker.gpu_input_types import ModelInputForGPUWithSamplingMetadata
from vllm.worker.gpu_input_builder import ModelInputForGPUBuilder
from vllm.worker.gpu_runner_base import GPUModelRunnerBase
from vllm.worker.cuda_graph_runner import CUDAGraphRunner 

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


logger = init_logger(__name__)


class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls = ModelInputForGPUWithSamplingMetadata
    _builder_cls = ModelInputForGPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        super().__init__(
            vllm_config,
            kv_cache_dtype,
            is_driver_worker,
            return_hidden_states,
            input_registry,
            mm_registry,
        )
        self.sampler = get_sampler()
        self.sampling_metadata_cache: Optional[SamplingMetadataCache] = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None
        self.builder = self._builder_cls(weakref.proxy(self))


    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input = \
            ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input_tensors = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        
        sampling_metadata = None
        if get_pp_group().is_last_rank:
            # get_generators is inherited from ModelRunnerBase (which GPUModelRunnerBase inherits from)
            generators = self.get_generators(finished_requests_ids) 
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, model_input_tensors.seq_lens,
                model_input_tensors.query_lens, self.device, self.pin_memory,
                generators, self.sampling_metadata_cache) # type: ignore[arg-type] # sampling_metadata_cache can be None
        
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)

        final_model_input = ModelInputForGPUWithSamplingMetadata(
            input_tokens=model_input_tensors.input_tokens,
            inputs_embeds=model_input_tensors.inputs_embeds,
            input_positions=model_input_tensors.input_positions,
            token_types=model_input_tensors.token_types,
            attn_metadata=model_input_tensors.attn_metadata,
            seq_lens=model_input_tensors.seq_lens,
            query_lens=model_input_tensors.query_lens,
            lora_mapping=model_input_tensors.lora_mapping,
            lora_requests=model_input_tensors.lora_requests,
            multi_modal_kwargs=model_input_tensors.multi_modal_kwargs,
            request_ids_to_seq_ids=model_input_tensors.request_ids_to_seq_ids,
            finished_requests_ids=model_input_tensors.finished_requests_ids,
            prompt_adapter_mapping=model_input_tensors.prompt_adapter_mapping,
            prompt_adapter_requests=model_input_tensors.prompt_adapter_requests,
            virtual_engine=virtual_engine, 
            async_callback=model_input_tensors.async_callback, 
            scheduler_outputs=model_input_tensors.scheduler_outputs, 
            previous_hidden_states=model_input_tensors.previous_hidden_states, 
            sampling_metadata=sampling_metadata, 
            is_prompt=is_prompt 
        )
        return final_model_input


    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor], 
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        if self.attn_state: 
            self.attn_state.begin_forward(model_input) # type: ignore[arg-type] # model_input is correct type

        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        
        current_stage_kv_caches: List[torch.Tensor] = kv_caches # Assuming kv_caches is already stage-specific List[Tensor]

        model_executable: Union[torch.nn.Module, CUDAGraphRunner] = self.model
        if prefill_meta is None and decode_meta and decode_meta.use_cuda_graph: 
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            use_inputs_embeds = model_input.inputs_embeds is not None
            
            graph_lookup_key = (graph_batch_size, use_inputs_embeds)
            
            if virtual_engine < len(self.graph_runners) and \
               graph_lookup_key in self.graph_runners[virtual_engine]:
                model_executable = self.graph_runners[virtual_engine][graph_lookup_key]
            else:
                logger.warning(f"CUDA graph not found for key {graph_lookup_key} on VE {virtual_engine}. Falling back to eager.")
                model_executable = self.model

            if previous_hidden_states is not None and isinstance(model_executable, CUDAGraphRunner): 
                if graph_batch_size > previous_hidden_states.shape[0]: # Pad only if needed
                    padding_size = graph_batch_size - previous_hidden_states.shape[0]
                    padding = torch.empty([padding_size, *previous_hidden_states.shape[1:]],
                                        dtype=previous_hidden_states.dtype,
                                        device=previous_hidden_states.device)
                    previous_hidden_states = torch.cat([previous_hidden_states, padding])
                elif graph_batch_size < previous_hidden_states.shape[0]: # Truncate if needed
                    previous_hidden_states = previous_hidden_states[:graph_batch_size]

        
        bypass_model_exec = False
        hidden_or_intermediate_states: Optional[Union[torch.Tensor, IntermediateTensors]] = None

        if self.need_recv_kv(model_input, current_stage_kv_caches): 
            (hidden_or_intermediate_states, bypass_model_exec, model_input # type: ignore
            ) = get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                model_executable, # type: ignore[arg-type]
                model_input,
                kv_caches=current_stage_kv_caches 
            )

        multi_modal_kwargs_val = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        
        model_call_kwargs: Dict[str, Any] = {
            "input_ids": model_input.input_tokens,
            "inputs_embeds": model_input.inputs_embeds,
            "positions": model_input.input_positions,
            # "kv_caches": current_stage_kv_caches, # Conditionally added below
            # "attn_metadata": model_input.attn_metadata, # Conditionally added below
            "intermediate_tensors": intermediate_tensors,
            **MultiModalKwargs.as_kwargs(multi_modal_kwargs_val, device=self.device),
            **seqlen_agnostic_kwargs,
        }
        if previous_hidden_states is not None:
            model_call_kwargs["previous_hidden_states"] = previous_hidden_states
        
        # Check model_executable's forward signature for token_types
        forward_params = inspect.signature(model_executable.forward if isinstance(model_executable, CUDAGraphRunner) else self.model.forward).parameters
        if "token_types" in forward_params:
            model_call_kwargs["token_types"] = model_input.token_types
        
        if isinstance(model_executable, CUDAGraphRunner):
            model_call_kwargs["kv_caches"] = current_stage_kv_caches
            model_call_kwargs["attn_metadata"] = model_input.attn_metadata

        model_forward_start_event: Optional[torch.cuda.Event] = None
        model_forward_end_event: Optional[torch.cuda.Event] = None

        # Assertion to check consistency before model execution
        if model_input.input_tokens is not None and model_input.attn_metadata is not None:
            num_meta_prefill = model_input.attn_metadata.num_prefill_tokens
            num_meta_decode = model_input.attn_metadata.num_decode_tokens
            assert model_input.input_tokens.shape[0] == num_meta_prefill + num_meta_decode, \
                f"Token count mismatch: input_tokens.shape[0]={model_input.input_tokens.shape[0]}, " \
                f"attn_metadata.num_prefill_tokens={num_meta_prefill}, " \
                f"attn_metadata.num_decode_tokens={num_meta_decode}"

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start_event = torch.cuda.Event(enable_timing=True)
            model_forward_end_event = torch.cuda.Event(enable_timing=True)
            model_forward_start_event.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                if isinstance(model_executable, CUDAGraphRunner):
                     graph_outputs = model_executable.forward(**model_call_kwargs) # type: ignore[arg-type]
                     if isinstance(graph_outputs, IntermediateTensors):
                         hidden_or_intermediate_states = graph_outputs
                     elif isinstance(graph_outputs, dict) and "hidden_states" in graph_outputs: 
                         hidden_or_intermediate_states = graph_outputs["hidden_states"] 
                else:
                    hidden_or_intermediate_states = model_executable(**model_call_kwargs) # type: ignore[operator]


        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time and model_forward_end_event):
            model_forward_end_event.record() 

        if self.need_send_kv(model_input, current_stage_kv_caches): 
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                model_executable, # type: ignore[arg-type]
                model_input,
                current_stage_kv_caches, 
                hidden_or_intermediate_states, 
            )

        if get_pp_group().is_last_rank:
            if self.return_hidden_states:
                if model_input.sampling_metadata is not None and model_input.sampling_metadata.seq_groups is not None:
                    for seq_group_metadata_item in model_input.sampling_metadata.seq_groups: 
                        seq_group_metadata_item.hidden_states = hidden_or_intermediate_states # type: ignore[assignment]
                return None 

            assert model_input.sampling_metadata is not None
            assert isinstance(hidden_or_intermediate_states, torch.Tensor), \
                f"Expected hidden_states to be a Tensor for sampling, got {type(hidden_or_intermediate_states)}"
            output: SamplerOutput = self.sampler(
                logits=hidden_or_intermediate_states,
                sampling_metadata=model_input.sampling_metadata,
            )
            return [output]
        else:
            assert isinstance(hidden_or_intermediate_states, IntermediateTensors), \
                 f"Expected IntermediateTensors for non-last PP rank, got {type(hidden_or_intermediate_states)}"
            return hidden_or_intermediate_states

    def need_recv_kv(self, model_input: ModelInputForGPUWithSamplingMetadata,
                     kv_caches: List[torch.Tensor]) -> bool:
        from vllm.distributed.kv_transfer.kv_transfer_state import has_kv_transfer_group # Moved import here
        if not has_kv_transfer_group():
            return False
        return (get_kv_transfer_group().has_recv_op(model_input, kv_caches) # type: ignore[arg-type]
                and not get_pp_group().is_first_rank)

    def need_send_kv(self, model_input: ModelInputForGPUWithSamplingMetadata,
                     kv_caches: List[torch.Tensor]) -> bool:
        from vllm.distributed.kv_transfer.kv_transfer_state import has_kv_transfer_group # Moved import here
        if not has_kv_transfer_group():
            return False
        return (get_kv_transfer_group().has_send_op(model_input, kv_caches) # type: ignore[arg-type]
                and not get_pp_group().is_last_rank)

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None: # type: ignore[override]
        # kv_caches is List[List[torch.Tensor]] for PP compatibility with base
        # but capture_model in GPUModelRunnerBase expects it to be stage specific.
        # We pass the CUDAGraphRunner class to the super method.
        # The super().capture_model will handle iterating through stages if necessary,
        # or we ensure it's called per stage.
        # For now, assume super().capture_model handles the List[List[torch.Tensor]] correctly
        # or expects to be called for a specific stage's kv_cache list.
        # The GPUModelRunnerBase.capture_model was modified to iterate stages.
        super().capture_model(kv_caches, CUDAGraphRunner)
