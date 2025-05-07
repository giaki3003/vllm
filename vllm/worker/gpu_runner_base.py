# SPDX-License-Identifier: Apache-2.0

import gc
import inspect
import itertools
import time
import weakref
from contextlib import contextmanager
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Generic) # Added TypeVar, Generic

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm.auto import tqdm # Added for capture_model progress bar

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.attention.backends.abstract import AttentionState
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             graph_capture)
from vllm.forward_context import set_forward_context # Added for capture_model
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping # Added for set_active_loras
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.layers import PromptAdapterMapping # Added for set_active_prompt_adapters
from vllm.prompt_adapter.worker_manager import (
    LRUCacheWorkerPromptAdapterManager)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, IntermediateTensors
from vllm.utils import (DeviceMemoryProfiler, GiB_bytes, PyObjectCache,
                        is_pin_memory_available, supports_dynamo)
from vllm.worker.model_runner_base import (
    InputProcessingError, ModelRunnerBase) # ModelRunnerInputBase removed as ModelInputForGPU is used
from vllm.worker.gpu_input_types import ModelInputForGPU
from vllm.worker.gpu_input_builder import ModelInputForGPUBuilder


if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend
    # CUDAGraphRunner will be defined in its own file later
    class CUDAGraphRunner(nn.Module): pass


logger = init_logger(__name__)

LORA_WARMUP_RANK = 8
_NUM_WARMUP_ITERS = 2

if supports_dynamo():
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 128

# Define the TypeVar for GPUModelRunnerBase
TGpuInput = TypeVar('TGpuInput', bound=ModelInputForGPU)

class GPUModelRunnerBase(ModelRunnerBase[TGpuInput], Generic[TGpuInput]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TGpuInput] # Use the TypeVar
    # _builder_cls will be set by subclasses if they use a specific builder
    # builder: ModelInputForGPUBuilder # This will be an instance variable set by subclass

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.max_batchsize_to_capture = \
            self.vllm_config.compilation_config.max_capture_size

        self.graph_runners: List[Dict[Tuple[int, bool], 'CUDAGraphRunner']] = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None

        self.has_inner_state = model_config.has_inner_state
        self.in_profile_run = False

        self.graph_block_tables = np.zeros(
            (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
            dtype=np.int32)

        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        ) if needs_attn_backend else None
        if self.attn_backend:
            self.attn_state: AttentionState = self.attn_backend.get_state_cls()(
                weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self)) # type: ignore[assignment] # CommonAttentionState is compatible

        self.input_registry = input_registry
        self.mm_registry = mm_registry

        self.model: nn.Module
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.prompt_adapter_manager: Optional[LRUCacheWorkerPromptAdapterManager] = None

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        self.inter_data_cache: Dict[int, PyObjectCache] = {}
        
        # The builder instance is typically initialized by the subclass (e.g., ModelRunner)
        # that knows which specific builder class to use.
        # For example, ModelRunner will initialize self.builder = ModelInputForGPUBuilder(weakref.proxy(self))
        # self.builder: ModelInputForGPUBuilder # Type hint for instance variable


    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler(self.device) as m:
            time_before_load = time.perf_counter()
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                assert supports_lora(
                    self.model
                ), f"{self.model.__class__.__name__} does not support LoRA yet."

                if supports_multimodal(self.model):
                    logger.warning(
                        "Regarding multimodal models, vLLM currently "
                        "only supports adding LoRA to language model.")

                text_config = self.model_config.hf_config.get_text_config()

                self.lora_manager = LRUCacheWorkerLoRAManager(
                    self.scheduler_config.max_num_seqs,
                    self.scheduler_config.max_num_batched_tokens,
                    self.vocab_size,
                    self.lora_config,
                    self.device,
                    self.model.embedding_modules,
                    self.model.embedding_padding_modules,
                    max_position_embeddings=text_config.
                    max_position_embeddings,
                )
                self.model = self.lora_manager.create_lora_manager(self.model)
            time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.device,
                self.prompt_adapter_config)
            self.model = (
                self.prompt_adapter_manager.create_prompt_adapter_manager(
                    self.model))

        if self.vllm_config.compilation_config.level ==\
            CompilationLevel.DYNAMO_AS_IS and supports_dynamo():
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config)
            self.model = torch.compile(
                self.model,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)

    def get_model(self) -> nn.Module:
        return self.model

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TGpuInput: # Return type uses the TypeVar
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.
        """
        assert hasattr(self, 'builder'), "Builder not initialized. Subclass should set it."
        
        self.builder.prepare(finished_requests_ids) # type: ignore[attr-defined] # builder is set by subclass
        for seq_group_metadata in seq_group_metadata_list:
            try:
                self.builder.add_seq_group(seq_group_metadata) # type: ignore[attr-defined]
            except Exception as e:
                raise InputProcessingError(seq_group_metadata.request_id,
                                           str(e)) from e

        self.builder.reset_cached_inter_data() # type: ignore[attr-defined]

        return self.builder.build() # type: ignore[attr-defined]

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        self._dummy_run(max_num_batched_tokens, max_num_seqs)

    def _add_dummy_loras(self, num_loras: int) -> list[LoRARequest]:
        assert num_loras > 0
        assert self.lora_manager is not None

        dummy_lora_requests: list[LoRARequest] = []
        with self.lora_manager.dummy_lora_cache():
            for idx in range(num_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
        return dummy_lora_requests

    def _remove_dummy_loras(self):
        assert self.lora_manager is not None
        self.remove_all_loras()

    def _dummy_run(self,
                   max_num_batched_tokens: int,
                   max_num_seqs: int = 1) -> None:
        with self.set_in_profile_run():
            sampling_params = \
                SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

            dummy_lora_requests_per_seq: List[LoRARequest] = []
            if self.lora_config:
                dummy_lora_requests = self._add_dummy_loras( # Ensure this is assigned
                    self.lora_config.max_loras)
                assert len(dummy_lora_requests) == self.lora_config.max_loras
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

            seqs: List[SequenceGroupMetadata] = []
            max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
                self.model_config)
            if max_mm_tokens > 0:
                max_num_seqs_orig = max_num_seqs
                max_num_seqs = min(max_num_seqs,
                                   max_num_batched_tokens // max_mm_tokens if max_mm_tokens > 0 else max_num_batched_tokens) # Avoid div by zero
                if max_num_seqs < 1:
                    expr = (f"min({max_num_seqs_orig}, "
                            f"{max_num_batched_tokens} // {max_mm_tokens})")
                    logger.warning(
                        "Computed max_num_seqs (%s) to be less than 1. "
                        "Setting it to the minimum value of 1.", expr)
                    max_num_seqs = 1
            
            current_batch_size = 0 # Renamed from batch_size to avoid confusion
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                current_batch_size += seq_len

                dummy_data = self.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                              seq_len,
                                              self.mm_registry)

                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    lora_request=dummy_lora_requests_per_seq[group_id]
                    if dummy_lora_requests_per_seq else None,
                    multi_modal_data=dummy_data.multi_modal_data,
                    multi_modal_placeholders=dummy_data.
                    multi_modal_placeholders,
                )
                seqs.append(seq)

            num_layers = self.model_config.get_num_layers(self.parallel_config)
            
            # kv_caches structure depends on PP size. For dummy run, assume single stage or adapt.
            # This is a simplified representation for the base class.
            # The actual kv_caches passed to execute_model will be more complex in PP.
            kv_caches_for_dummy: List[torch.Tensor] = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]

            finished_requests_ids = [seq.request_id for seq in seqs]
            
            # This will call the prepare_model_input of the concrete subclass (e.g., ModelRunner)
            # which returns the appropriate model input type (e.g., ModelInputForGPUWithSamplingMetadata)
            model_input_for_exec = self.prepare_model_input(
                seqs, finished_requests_ids=finished_requests_ids
            ) # virtual_engine defaults to 0 in prepare_model_input if not specified by subclass

            intermediate_tensors = None
            if self.parallel_config.pipeline_parallel_size > 1 and not get_pp_group().is_first_rank :
                intermediate_tensors = \
                    self.model.make_empty_intermediate_tensors(
                    batch_size=current_batch_size,
                    dtype=self.model_config.dtype,
                    device=self.device)

            if model_input_for_exec.attn_metadata is not None:
                model_input_for_exec.attn_metadata.enable_kv_scales_calculation = False
            
            # This will call the execute_model of the concrete subclass
            self.execute_model(
                model_input_for_exec,
                kv_caches=kv_caches_for_dummy,
                intermediate_tensors=intermediate_tensors
            )

            torch.cuda.synchronize()
            if self.lora_config:
                self._remove_dummy_loras()

            gc.collect()
            torch.cuda.synchronize()
            return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: "LoRAMapping") -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()

    def remove_all_prompt_adapters(self):
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.remove_all_adapters()

    def set_active_prompt_adapters(
            self, prompt_adapter_requests: Set[PromptAdapterRequest],
            prompt_adapter_mapping: "PromptAdapterMapping") -> None:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.set_active_adapters(
            prompt_adapter_requests, prompt_adapter_mapping)

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.add_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.remove_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.pin_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.list_adapters()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]], CUDAGraphRunnerClass: Type['CUDAGraphRunner']) -> None:
        # CUDAGraphRunnerClass is passed to avoid circular import at module level
        assert not self.model_config.enforce_eager
        logger.info("Capturing cudagraphs for decoding...")
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
        input_positions = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
        inputs_embeds = torch.zeros(
            (max_batch_size, self.model_config.get_hidden_size()),
            dtype=self.model_config.dtype,
            device=self.device)
        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions, (3, 1)).cuda(device=self.device)
        
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [max_batch_size, self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device)

        intermediate_inputs = None
        if self.parallel_config.pipeline_parallel_size > 1 and not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        dummy_lora_id: Optional[int] = None
        dummy_lora_request_set: Set[LoRARequest] = set() # Use set for set_active_loras
        if self.lora_config:
            dummy_lora_requests_list = self._add_dummy_loras(num_loras=1)
            if dummy_lora_requests_list:
                 dummy_lora_request_set.add(dummy_lora_requests_list[0])
                 dummy_lora_id = dummy_lora_requests_list[0].lora_int_id


        with self.attn_state.graph_capture(max_batch_size), graph_capture(self.device) as graph_capture_context:
            stages_to_capture = []
            if self.parallel_config.pipeline_parallel_size > 1:
                stages_to_capture.append(get_pp_group().rank_in_group)
            else:
                stages_to_capture.append(0)

            for virtual_engine in stages_to_capture:
                cudagraph_capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cudagraph_inputs_embeds = ((True, False) if self.model_config.enable_prompt_embeds else (False,))
                
                compilation_cases = list(itertools.product(cudagraph_capture_sizes, cudagraph_inputs_embeds))
                
                if get_tensor_model_parallel_rank() == 0:
                    compilation_cases = tqdm(compilation_cases, desc="Capturing CUDA graph shapes")

                for current_capture_batch_size, use_inputs_embeds in compilation_cases: # Renamed batch_size
                    attn_metadata = self.attn_state.graph_capture_get_metadata_for_batch(
                        current_capture_batch_size, # Use renamed variable
                        is_encoder_decoder_model=self.model_config.is_encoder_decoder
                    )
                    if attn_metadata: # Ensure attn_metadata is not None
                        attn_metadata.enable_kv_scales_calculation = False

                    if self.lora_config and dummy_lora_id is not None:
                        lora_mapping = LoRAMapping(
                            index_mapping=[dummy_lora_id] * current_capture_batch_size, # Use renamed
                            prompt_mapping=[dummy_lora_id] * current_capture_batch_size, # Use renamed
                            is_prefill=False)
                        self.set_active_loras(dummy_lora_request_set, lora_mapping)

                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * current_capture_batch_size, # Use renamed
                            [-1] * current_capture_batch_size, # Use renamed
                        )
                        self.set_active_prompt_adapters(set(), prompt_adapter_mapping)
                    
                    graph_runner = CUDAGraphRunnerClass( # Use passed class
                        self.model, 
                        self.attn_backend.get_name() if self.attn_backend else "no_attn_backend", # Handle None attn_backend
                        self.attn_state.graph_clone(current_capture_batch_size), # Use renamed
                        self.model_config.is_encoder_decoder
                    )

                    current_kv_caches = kv_caches[virtual_engine] if virtual_engine < len(kv_caches) else []


                    capture_inputs_dict = { # Renamed to avoid conflict
                        "input_ids": input_tokens[:current_capture_batch_size], # Use renamed
                        "inputs_embeds": inputs_embeds[:current_capture_batch_size] if use_inputs_embeds else None, # Use renamed
                        "positions": input_positions[..., :current_capture_batch_size], # Use renamed
                        "intermediate_tensors": intermediate_inputs[:current_capture_batch_size] if intermediate_inputs is not None else None, # Use renamed, key fixed
                        "kv_caches": current_kv_caches,
                        "attn_metadata": attn_metadata,
                        "memory_pool": self.graph_memory_pool,
                        "stream": graph_capture_context.stream
                    }
                    if previous_hidden_states is not None:
                        capture_inputs_dict["previous_hidden_states"] = previous_hidden_states[:current_capture_batch_size] # Use renamed

                    if self.has_inner_state:
                        capture_inputs_dict.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(current_capture_batch_size) # Use renamed
                        })
                    if self.model_config.is_encoder_decoder:
                        self._update_inputs_to_capture_for_enc_dec_model(capture_inputs_dict)

                    with set_forward_context(attn_metadata, self.vllm_config, virtual_engine):
                        graph_runner.capture(**capture_inputs_dict)
                    
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][(current_capture_batch_size, use_inputs_embeds)] = graph_runner # Use renamed

        if self.lora_config:
            self._remove_dummy_loras()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / GiB_bytes)


    def _update_inputs_to_capture_for_enc_dec_model(self,
                                                    capture_inputs: Dict[str,
                                                                         Any]):
        capture_inputs["encoder_input_ids"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)
        capture_inputs["encoder_positions"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()