# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import inspect
import itertools
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm.auto import tqdm

import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionState
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import CompilationLevel, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             graph_capture)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import (Sampler, SamplerOutput,
                                                get_sampler)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.worker_manager import (
    LRUCacheWorkerPromptAdapterManager)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import (DeviceMemoryProfiler, GiB_bytes, PyObjectCache,
                        async_tensor_h2d, flatten_2d_lists,
                        is_pin_memory_available, supports_dynamo,
                        weak_ref_tensor)
from vllm.worker.model_runner_base import (
    InputProcessingError, ModelRunnerBase, ModelRunnerInputBase,
    ModelRunnerInputBuilderBase, _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)
from vllm.worker.gpu_model_inputs import (ModelInputForGPU, ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata, TModelInputForGPU)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8

_NUM_WARMUP_ITERS = 2

# For now, bump up cache limits for recompilations during CUDA graph warmups.
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 128


class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForGPU]
    _builder_cls: Type[ModelInputForGPUBuilder]
    builder: ModelInputForGPUBuilder

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

        #
        self.graph_runners: List[Dict[Tuple[int, bool], CUDAGraphRunner]] = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        # Defensively check if PP group is initialized before trying to get rank for logging
        pp_rank_for_log = 'N/A_INIT'
        if self.parallel_config.pipeline_parallel_size > 1:
            try:
                # Attempt to get rank only if group might be initialized
                if torch.distributed.is_initialized() and get_pp_group() is not None: # False to not assert
                    pp_rank_for_log = get_pp_group().rank_in_group
                else:
                    pp_rank_for_log = 'N/A_NO_GROUP'
            except AssertionError: # Catch if get_pp_group asserts
                 pp_rank_for_log = 'N/A_ASSERT'
        else:
            pp_rank_for_log = 0
        print(f"[DEBUG LIFECYCLE] GPUModelRunnerBase __init__ for self ID {id(self)}, Configured PP Size {self.parallel_config.pipeline_parallel_size}, Logged PP Rank {pp_rank_for_log}. Initialized self.graph_runners with ID: {id(self.graph_runners)} and size {len(self.graph_runners)}")
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.

        self.has_inner_state = model_config.has_inner_state

        self.in_profile_run = False

        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max seq len to capture / block size).
        self.graph_block_tables = np.zeros(
            (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
            dtype=np.int32)
        # Defensively check if PP group is initialized before trying to get rank for logging
        pp_rank_for_log = 'N/A_INIT'
        if self.parallel_config.pipeline_parallel_size > 1:
            try:
                # Attempt to get rank only if group might be initialized
                if torch.distributed.is_initialized() and get_pp_group() is not None: # False to not assert
                    pp_rank_for_log = get_pp_group().rank_in_group
                else:
                    pp_rank_for_log = 'N/A_NO_GROUP'
            except AssertionError: # Catch if get_pp_group asserts
                 pp_rank_for_log = 'N/A_ASSERT'
        else:
            pp_rank_for_log = 0
        graph_runners_id = id(self.graph_runners)
        graph_runners_len = len(self.graph_runners)
        ve0_id_log = id(self.graph_runners[0]) if graph_runners_len > 0 and self.graph_runners[0] is not None else 'N/A'
        ve1_id_log = id(self.graph_runners[1]) if graph_runners_len > 1 and self.graph_runners[1] is not None else 'N/A'
        print(f"[DEBUG LIFECYCLE] GPUModelRunnerBase __init__ for self ID {id(self)}, Configured PP Size {self.parallel_config.pipeline_parallel_size}, Logged PP Rank {pp_rank_for_log}. Initialized self.graph_runners with ID: {graph_runners_id}, size {graph_runners_len}, VE0 ID: {ve0_id_log}, VE1 ID: {ve1_id_log}")

        # Attention-free but stateful models like Mamba need a placeholder attn
        # backend, as the attention metadata is needed to manage internal state.
        # However we must bypass attention selection altogether for some models
        # used for speculative decoding to avoid a divide-by-zero in
        # model_config.get_head_size()
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
            self.attn_state = self.attn_backend.get_state_cls()(
                weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.prompt_adapter_manager: LRUCacheWorkerPromptAdapterManager = None
        self.sampler = get_sampler()

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceGroupToSample object. In Pipeline-Parallel, we have
        # more than 1 Scheduler, resulting in a potential back-to-back
        # prepare_model_inputs() call. This clobbers the cached
        # SequenceGroupToSample objects, as we reset the cache during
        # every prepare_model_inputs() call.
        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        # Corrected rank_in_group and added more ID details
        gr_id = id(self.graph_runners) if hasattr(self, 'graph_runners') else 'N/A'
        gr_len = len(self.graph_runners) if hasattr(self, 'graph_runners') and self.graph_runners is not None else -1
        ve0_id_log = id(self.graph_runners[0]) if hasattr(self, 'graph_runners') and self.graph_runners is not None and gr_len > 0 and self.graph_runners[0] is not None else 'N/A'
        ve1_id_log = id(self.graph_runners[1]) if hasattr(self, 'graph_runners') and self.graph_runners is not None and gr_len > 1 and self.graph_runners[1] is not None else 'N/A'
        pp_rank_log = 'N/A_EARLY'
        if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized():
            pp_group_val = get_pp_group()
            if pp_group_val is not None:
                pp_rank_log = pp_group_val.rank_in_group
        elif self.parallel_config.pipeline_parallel_size <= 1:
            pp_rank_log = 0
        print(f"[DEBUG LIFECYCLE] ENTERING load_model for self ID {id(self)}, PP Rank {pp_rank_log}. ID of self.graph_runners: {gr_id}, VE0 ID: {ve0_id_log}, VE1 ID: {ve1_id_log}")
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

                # Use get_text_config() in case of multimodal models
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
    ) -> TModelInputForGPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        self.builder.prepare(finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            try:
                self.builder.add_seq_group(seq_group_metadata)
            except Exception as e:
                # Raise an exception that tracks the ID of the bad request
                raise InputProcessingError(seq_group_metadata.request_id,
                                           str(e)) from e

        self.builder.reset_cached_inter_data()

        return self.builder.build()  # type: ignore

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
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: In profile_run, about to call _dummy_run.")
        max_num_seqs = self.scheduler_config.max_num_seqs
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: In profile_run, _dummy_run completed.")
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
        # Remove dummy loras.
        assert self.lora_manager is not None
        self.remove_all_loras()

    def _dummy_run(self,
                   max_num_batched_tokens: int,
                   max_num_seqs: int = 1) -> None:
        with self.set_in_profile_run():
            # Enable top-k sampling to reflect the accurate memory usage.
            sampling_params = \
                SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

            # This represents the maximum number of different requests
            # that will have unique loras, and therefore the max amount of
            # memory consumption. Create dummy lora request copies from the
            # lora request passed in, which contains a lora from the lora
            # warmup path.
            dummy_lora_requests: List[LoRARequest] = []
            dummy_lora_requests_per_seq: List[LoRARequest] = []
            if self.lora_config:
                dummy_lora_requests = self._add_dummy_loras(
                    self.lora_config.max_loras)
                assert len(dummy_lora_requests) == self.lora_config.max_loras
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

            # Profile memory usage with max_num_sequences sequences and the
            # total number of tokens equal to max_num_batched_tokens.
            seqs: List[SequenceGroupMetadata] = []
            # Additional GPU memory may be needed for multi-modal encoding,
            # which needs to be accounted for when calculating the GPU blocks
            # for vLLM blocker manager.
            # To exercise the worst scenario for GPU memory consumption,
            # the number of seqs (batch_size) is chosen to maximize the number
            # of images processed.

            max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
                self.model_config)
            if max_mm_tokens > 0:
                max_num_seqs_orig = max_num_seqs
                max_num_seqs = min(max_num_seqs,
                                   max_num_batched_tokens // max_mm_tokens)
                if max_num_seqs < 1:
                    expr = (f"min({max_num_seqs_orig}, "
                            f"{max_num_batched_tokens} // {max_mm_tokens})")
                    logger.warning(
                        "Computed max_num_seqs (%s) to be less than 1. "
                        "Setting it to the minimum value of 1.", expr)
                    max_num_seqs = 1

            batch_size = 0
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                batch_size += seq_len

                dummy_data = self.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                              seq_len,
                                              self.mm_registry)
                #logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: _dummy_run using seq_len={seq_len}, max_batched_tokens={max_num_batched_tokens}, max_seqs={max_num_seqs})")

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

            # Run the model with the dummy inputs.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            # use an empty tensor instead of `None`` to force Dynamo to pass
            # it by reference, rather by specializing on the value ``None``.
            # the `dtype` argument does not matter, and we use `float32` as
            # a placeholder (it has wide hardware support).
            # it is important to create tensors inside the loop, rather than
            # multiplying the list, to avoid Dynamo from treating them as
            # tensor aliasing.
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            finished_requests_ids = [seq.request_id for seq in seqs]
            model_input = self.prepare_model_input(
                seqs, finished_requests_ids=finished_requests_ids)
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = \
                    self.model.make_empty_intermediate_tensors(
                    batch_size=batch_size,
                    dtype=self.model_config.dtype,
                    device=self.device)

            # Disable KV Scale Calculation for dummy data during profile run
            if model_input.attn_metadata is not None:
                model_input.attn_metadata.enable_kv_scales_calculation = False

            logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: _dummy_run before synchronize() line 1419.")
            self.execute_model(model_input, kv_caches, intermediate_tensors)
            logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: _dummy_run after synchronize() line 1419.")
            torch.cuda.synchronize()
            if self.lora_config:
                self._remove_dummy_loras()

            logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: At end of _dummy_run, before final gc.collect and synchronize.")
            gc.collect() # Temporarily commented out for debugging hang at end of _dummy_run
            logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: gc.collect() at end of _dummy_run SKIPPED.")
            torch.cuda.synchronize()
            logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.vllm_config.parallel_config.rank if self.vllm_config else 'N/A'}: At end of _dummy_run, after final gc.collect and synchronize.")
            return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
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
            prompt_adapter_mapping: PromptAdapterMapping) -> None:
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
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        # Corrected rank_in_group and added more ID details
        gr_len_capture = len(self.graph_runners) if self.graph_runners is not None else 0
        ve0_id_log_capture = id(self.graph_runners[0]) if self.graph_runners is not None and gr_len_capture > 0 and self.graph_runners[0] is not None else 'N/A'
        ve1_id_log_capture = id(self.graph_runners[1]) if self.graph_runners is not None and gr_len_capture > 1 and self.graph_runners[1] is not None else 'N/A'
        pp_rank_log_capture = 'N/A_CAPTURE_ENTRY'
        if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized():
            pp_group_val_capture = get_pp_group()
            if pp_group_val_capture is not None:
                pp_rank_log_capture = pp_group_val_capture.rank_in_group
        elif self.parallel_config.pipeline_parallel_size <=1:
             pp_rank_log_capture = 0
        print(f"[DEBUG LIFECYCLE] Worker rank {get_tensor_model_parallel_rank()}, PP Rank {pp_rank_log_capture}: ENTERING capture_model. ID of self: {id(self)}, ID of self.graph_runners: {id(self.graph_runners)}, VE0 ID: {ve0_id_log_capture}, VE1 ID: {ve1_id_log_capture}")
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing cudagraphs for decoding. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI. "
                    "If out-of-memory error occurs during cudagraph capture,"
                    " consider decreasing `gpu_memory_utilization` or "
                    "switching to eager mode. You can also reduce the "
                    "`max_num_seqs` as needed to decrease memory usage.")
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros(max_batch_size,
                                   dtype=torch.long,
                                   device=self.device)
        input_positions = torch.zeros(max_batch_size,
                                      dtype=torch.long,
                                      device=self.device)
        inputs_embeds = torch.zeros(
            (max_batch_size, self.model_config.get_hidden_size()),
            dtype=self.model_config.dtype,
            device=self.device)
        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions,
                                         (3, 1)).cuda(device=self.device)
        # Prepare dummy previous_hidden_states only if needed by the model.
        # This is used by draft models such as EAGLE.
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(
                self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [max_batch_size,
                 self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device)

        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        dummy_lora_id: Optional[int] = None
        dummy_lora_request: LoRARequest = []
        if self.lora_config:
            # The goal is to capture the LoRA kernels in cuda graphs.
            # for this purpose, as single dummy lora is sufficient.
            dummy_lora_requests = self._add_dummy_loras(num_loras=1)
            assert len(dummy_lora_requests) == 1
            dummy_lora_request = dummy_lora_requests[0]
            dummy_lora_id = dummy_lora_request.lora_int_id

        with self.attn_state.graph_capture(max_batch_size), graph_capture(
                self.device) as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.

            # If pipeline parallel > 1, only capture for the current worker's stage.
            # The worker's self.model and self.attn_state are for its specific stage.
            # kv_caches passed from worker is also structured for its stage at its pp_rank.
            current_worker_pp_rank = get_pp_group().rank_in_group # Assumes get_pp_group is available
            
            stages_to_capture = []
            if self.parallel_config.pipeline_parallel_size > 1:
                stages_to_capture.append(current_worker_pp_rank)
                # Also, the first stage (rank 0) might be special if it handles embeddings
                # and the last stage for lm_head, but CUDAGraphRunner uses self.model which is stage-specific.
                # So, only capturing current worker's rank seems correct.
                # The original code iterated all virtual_engines, which would try to use this worker's
                # model parts with KV caches from other (None for this worker) stages.
            else: # Not pipelined, so only one "stage" (virtual_engine 0)
                stages_to_capture.append(0)

            print(f"[DEBUG TYPE CHECK] type(range) before problematic print: {type(range)}")
            print(f"[DEBUG TYPE CHECK] type(get_pp_group().rank_in_group) before problematic print: {type(get_pp_group().rank_in_group)}")
            print(f"[DEBUG STAGES] In capture_model for worker rank {get_tensor_model_parallel_rank()}, current_worker_pp_rank: {get_pp_group().rank_in_group}, calculated stages_to_capture: {stages_to_capture}, total VEs: {list(range(self.parallel_config.pipeline_parallel_size))}")
            for virtual_engine in stages_to_capture:
                # Corrected rank_in_group and added more ID details
                ve_dict_id_proc = id(self.graph_runners[virtual_engine]) if virtual_engine < len(self.graph_runners) and self.graph_runners[virtual_engine] is not None else 'N/A'
                pp_rank_log_ve_proc = get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized() and get_pp_group() is not None else (0 if self.parallel_config.pipeline_parallel_size <=1 else 'N/A_VE_PROC')
                print(f"[DEBUG CAPTURE_MODEL] VE {virtual_engine} (PP Rank {pp_rank_log_ve_proc}): Processing. Dict ID for this VE: {ve_dict_id_proc}. Current keys: {list(self.graph_runners[virtual_engine].keys()) if virtual_engine < len(self.graph_runners) and self.graph_runners[virtual_engine] is not None else 'N/A or new'}")
                # We need to not only iterate over batch sizes, but also whether
                # to use inputs_embeds or not, hence we use the cartesian
                # product.
                cudagraph_capture_sizes = self.vllm_config.compilation_config\
                    .cudagraph_capture_sizes
                cudagraph_inputs_embeds = ((
                    True, False) if self.model_config.enable_prompt_embeds else
                                           (False, ))
                compilation_cases = itertools.product(
                    cudagraph_capture_sizes,
                    cudagraph_inputs_embeds,
                )
                # Only rank 0 should print progress bar during capture
                if get_tensor_model_parallel_rank() == 0:
                    compilation_cases = tqdm(
                        list(compilation_cases),
                        desc="Capturing CUDA graph shapes")
                for batch_size, use_inputs_embeds in compilation_cases:
                    print(f"[DEBUG _CAPTURE_CUDA_GRAPH] Worker rank {get_tensor_model_parallel_rank()}, PP Rank {get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 else 0}, VE {virtual_engine}: Iterating compilation_case: batch_size={batch_size}, use_inputs_embeds={use_inputs_embeds}")
                    attn_metadata = (
                        self.attn_state.graph_capture_get_metadata_for_batch(
                            batch_size,
                            is_encoder_decoder_model=self.model_config.
                            is_encoder_decoder))
                    # Disable KV Scale Calculation for graph capture
                    attn_metadata.enable_kv_scales_calculation = False

                    # # Check if KV cache for this stage is valid for decode capture
                    # current_stage_kv_cache_list = kv_caches[virtual_engine] # This is List[Tensor_5D] or None
                    #
                    # # Note: attn_metadata.decode_metadata is a property that computes based on num_decode_tokens > 0
                    # # We are primarily concerned with decode captures.
                    # # A more direct check might be if this batch_size implies a decode-only scenario.
                    # # For simplicity, if decode_metadata would be true AND cache is bad, skip.
                    # # The graph capture is primarily for decode, so if decode_metadata is None, it's likely a prefill-like capture.
                    #
                    # is_decode_capture_attempt = False
                    # if attn_metadata.num_decode_tokens > 0 and attn_metadata.num_prefills == 0:
                    #     is_decode_capture_attempt = True
                    #
                    # if is_decode_capture_attempt and \
                    #    (current_stage_kv_cache_list is None or \
                    #     not all(t.numel() > 0 for t in current_stage_kv_cache_list if t is not None)):
                    #     logger.info(f"Skipping CUDAGraph capture for decode on VE {virtual_engine}, "
                    #                 f"batch_size {batch_size}, use_inputs_embeds {use_inputs_embeds} "
                    #                 f"due to empty or None KV cache for this stage.")
                    #     continue # Skip this specific capture case

                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(index_mapping=[dummy_lora_id] * batch_size,
                                   prompt_mapping=[dummy_lora_id] * batch_size,
                                   is_prefill=False))
                        self.set_active_loras(set([dummy_lora_request]),
                                              lora_mapping)

                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * batch_size,
                            [-1] * batch_size,
                        )
                        self.set_active_prompt_adapters(
                            set(), prompt_adapter_mapping)
                    graph_runner = CUDAGraphRunner(
                        self.model, self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder)

                    capture_inputs = {
                        "input_ids":
                        input_tokens[:batch_size],
                        "inputs_embeds":
                        inputs_embeds[:batch_size]
                        if use_inputs_embeds else None,
                        "positions":
                        input_positions[..., :batch_size],
                        "intermediate_inputs":
                        intermediate_inputs[:batch_size]
                        if intermediate_inputs is not None else None,
                        "kv_caches":
                        kv_caches[virtual_engine],
                        "attn_metadata":
                        attn_metadata,
                        "memory_pool":
                        self.graph_memory_pool,
                        "stream":
                        graph_capture_context.stream
                    }
                    if previous_hidden_states is not None:
                        capture_inputs[
                            "previous_hidden_states"] = previous_hidden_states[:
                                                                               batch_size]

                    if self.has_inner_state:
                        # Only used by Mamba-based models CUDA graph atm (Jamba)
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(
                                batch_size)
                        })
                    if self.model_config.is_encoder_decoder:
                        # add the additional inputs to capture for
                        # encoder-decoder models.
                        self._update_inputs_to_capture_for_enc_dec_model(
                            capture_inputs)

                    with set_forward_context(attn_metadata, self.vllm_config,
                                             virtual_engine):
                        print(f"[DEBUG CAPTURE] Worker rank {get_tensor_model_parallel_rank()}, PP Rank {get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 else 0}, VE {virtual_engine}: About to call graph_runner.capture for key ({batch_size}, {use_inputs_embeds})")
                        graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][(
                        batch_size, use_inputs_embeds)] = graph_runner
                    # Corrected rank_in_group
                    pp_rank_log_assign = get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized() and get_pp_group() is not None else (0 if self.parallel_config.pipeline_parallel_size <=1 else 'N/A_ASSIGN')
                    print(f"[DEBUG CAPTURE_ASSIGN] Worker rank {get_tensor_model_parallel_rank()}, PP Rank {pp_rank_log_assign}, VE {virtual_engine}: Assigned graph for key ({batch_size}, {use_inputs_embeds}). Dict ID for this VE: {id(self.graph_runners[virtual_engine])}. Current VE keys: {list(self.graph_runners[virtual_engine].keys())}")

        if self.lora_config:
            self._remove_dummy_loras()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / GiB_bytes)
        # Corrected rank_in_group and removed duplicate
        gr_len_exit = len(self.graph_runners)
        ve0_id_log_exit = id(self.graph_runners[0]) if gr_len_exit > 0 and self.graph_runners[0] is not None else 'N/A'
        ve0_keys_log_exit = list(self.graph_runners[0].keys()) if gr_len_exit > 0 and self.graph_runners[0] is not None else 'VE 0 empty/None/N/A'
        ve1_id_log_exit = id(self.graph_runners[1]) if gr_len_exit > 1 and self.graph_runners[1] is not None else 'N/A'
        ve1_keys_log_exit = list(self.graph_runners[1].keys()) if gr_len_exit > 1 and self.graph_runners[1] is not None else 'VE 1 empty/None/N/A'
        pp_rank_log_exit = get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized() and get_pp_group() is not None else (0 if self.parallel_config.pipeline_parallel_size <=1 else 'N/A_EXIT_CAPTURE')
        print(f"[DEBUG LIFECYCLE] Worker rank {get_tensor_model_parallel_rank()}, PP Rank {pp_rank_log_exit}: EXITING capture_model. ID of self.graph_runners: {id(self.graph_runners)}, VE0 ID: {ve0_id_log_exit}, Keys VE0: {ve0_keys_log_exit}, VE1 ID: {ve1_id_log_exit}, Keys VE1: {ve1_keys_log_exit}")

    def _update_inputs_to_capture_for_enc_dec_model(self,
                                                    capture_inputs: Dict[str,
                                                                         Any]):
        """
        Updates the set of input tensors needed for CUDA graph capture in an
        encoder-decoder model.

        This method modifies the provided `capture_inputs` dictionary by
        adding tensors specific to encoder-decoder specific models that
        need to be captured for CUDA Graph replay.
        """
        # During the decode phase encoder_input_ids and encoder_positions are
        # unset. Do the same thing for graph capture.
        capture_inputs["encoder_input_ids"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)
        capture_inputs["encoder_positions"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

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
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, model_input.seq_lens,
                model_input.query_lens, self.device, self.pin_memory,
                generators, self.sampling_metadata_cache)
        else:
            sampling_metadata = None
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

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

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            use_inputs_embeds = model_input.inputs_embeds is not None
            # ---- START DEBUG PRINTS for execute_model ----
            # This print was already added in a previous step, ensuring it's correctly placed / not duplicated.
            # Corrected rank_in_group and added more ID details
            ve_dict_id_exec = id(self.graph_runners[virtual_engine]) if virtual_engine < len(self.graph_runners) and self.graph_runners[virtual_engine] is not None else 'N/A'
            pp_rank_log_exec = get_pp_group().rank_in_group if self.parallel_config.pipeline_parallel_size > 1 and torch.distributed.is_initialized() and get_pp_group() is not None else (0 if self.parallel_config.pipeline_parallel_size <=1 else 'N/A_EXEC')
            print(f"[DEBUG LIFECYCLE] In execute_model, Worker rank {get_tensor_model_parallel_rank()}, PP Rank {pp_rank_log_exec}. ID of self: {id(self)}, ID of self.graph_runners: {id(self.graph_runners)}, ID of self.graph_runners[{virtual_engine}]: {ve_dict_id_exec}")
            graph_lookup_key = (graph_batch_size, use_inputs_embeds)
            print(f"[DEBUG EXECUTE] In execute_model, virtual_engine: {virtual_engine}") # This was already present
            print(f"[DEBUG EXECUTE] Looking up graph with key: {graph_lookup_key}") # This was already present
            if virtual_engine in self.graph_runners: # This was already present
                print(f"[DEBUG EXECUTE] Available keys for VE {virtual_engine}: {list(self.graph_runners[virtual_engine].keys())}")
            else:
                print(f"[DEBUG EXECUTE] No graphs captured for VE {virtual_engine} at all.")
            # ---- END DEBUG PRINTS for execute_model ----
            model_executable = self.graph_runners[virtual_engine][(
                graph_batch_size, use_inputs_embeds)]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    inputs_embeds=model_input.inputs_embeds,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                 device=self.device),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_input.async_callback(model_forward_start, model_forward_end)

        # Sample the next token.
        if get_pp_group().is_last_rank:
            output: SamplerOutput = self.sampler(
                hidden_or_intermediate_states,
                model_input.sampling_metadata,
            )
        else:
            output = hidden_or_intermediate_states
        return output

    def need_recv_kv(self, model_input, kv_caches) -> bool:
        """
        Whether the current worker needs to receive KV caches from other workers.
        """
        # If KV cache transfer is not enabled, no need to receive KV caches.
        if not (self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.enable):
            return False

        # If the current worker is the first worker in the pipeline, no need to
        # receive KV caches.
        if get_pp_group().is_first_rank:
            return False

        # If the current worker is not the driver worker, no need to receive KV
        # caches.
        if not self.is_driver_worker:
            return False

        # If the current worker is the driver worker and it is not the first
        # worker in the pipeline, it needs to receive KV caches.
        return True

    def need_send_kv(self, model_input, kv_caches) -> bool:
        """
        Whether the current worker needs to send KV caches to other workers.
        """
        # If KV cache transfer is not enabled, no need to send KV caches.
        if not (self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.enable):
            return False

        # If the current worker is the last worker in the pipeline, no need to
        # send KV caches.
        if get_pp_group().is_last_rank:
            return False

        # If the current worker is not the driver worker, no need to send KV
        # caches.
        if not self.is_driver_worker:
            return False

        # If the current worker is the driver worker and it is not the last
        # worker in the pipeline, it needs to send KV caches.
        return True


class CUDAGraphRunner(nn.Module):
    """A class that runs a model with CUDA graph.

    The CUDAGraphRunner is responsible for capturing the CUDA graph of a model
    and replaying it. It is used to optimize the execution of the model by
    reducing the overhead of launching kernels.
    """

    def __init__(self, model: nn.Module, backend_name: str,
                 attn_state: AttentionState, is_encoder_decoder_model: bool):
        super().__init__()
        self.model = model
        self.backend_name = backend_name
        self.attn_state = attn_state
        self.is_encoder_decoder_model = is_encoder_decoder_model
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    @property
    def graph_name(self) -> str:
        return f"{self.model.__class__.__name__}_{self.backend_name}"

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_inputs: Optional[IntermediateTensors] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        seqlen_agnostic_capture_inputs: Optional[Dict[str, Any]] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Captures the CUDA graph of the model.

        Args:
            input_ids: The input token IDs.
            positions: The positions of the input tokens.
            kv_caches: The KV caches.
            attn_metadata: The attention metadata.
            memory_pool: The memory pool to use for the CUDA graph.
            stream: The CUDA stream to use for capturing the graph.
            inputs_embeds: The input embeddings. If not None, input_ids will be
                ignored.
            intermediate_inputs: The intermediate inputs from the previous pipeline
                stage.
            previous_hidden_states: The hidden states from the previous step.
                Used by draft models such as EAGLE.
            seqlen_agnostic_capture_inputs: The sequence length agnostic inputs.
                Used by Mamba-based models.
            encoder_input_ids: The input token IDs for the encoder.
            encoder_positions: The positions of the input tokens for the encoder.
        """
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initializations (e.g., allocating KV caches).
        with set_forward_context(attn_metadata, None, 0):
            self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_inputs,
                previous_hidden_states=previous_hidden_states,
                **(seqlen_agnostic_capture_inputs or {}),
                encoder_input_ids=encoder_input_ids,
                encoder_positions=encoder_positions,
            )
        torch.cuda.synchronize(stream=stream)

        # Capture the graph.
        self.graph = torch.cuda.CUDAGraph()
        if memory_pool is None:
            memory_pool = self.graph.pool()
        with torch.cuda.graph(self.graph, pool=memory_pool, stream=stream):
            with set_forward_context(attn_metadata, None, 0):
                output_hidden_or_intermediate_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=kv_caches,
                    attn_metadata=attn_metadata,
                    inputs_embeds=inputs_embeds,
                    intermediate_tensors=intermediate_inputs,
                    previous_hidden_states=previous_hidden_states,
                    **(seqlen_agnostic_capture_inputs or {}),
                    encoder_input_ids=encoder_input_ids,
                    encoder_positions=encoder_positions,
                )
                if isinstance(output_hidden_or_intermediate_states,
                              IntermediateTensors):
                    # For pipeline parallel, the graph output is a dictionary of
                    # tensors.
                    hidden_or_intermediate_states = IntermediateTensors(
                        tensors={
                            k: v.clone()
                            for k, v in output_hidden_or_intermediate_states.
                            items()
                        })
                else:
                    # For tensor parallel, the graph output is a single tensor.
                    hidden_or_intermediate_states = \
                        output_hidden_or_intermediate_states.clone()
        torch.cuda.synchronize(stream=stream)

        # Save the input and output buffers.
        if inputs_embeds is not None:
            self.input_buffers["inputs_embeds"] = inputs_embeds
        else:
            self.input_buffers["input_ids"] = input_ids
        self.input_buffers["positions"] = positions
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.items())
        if previous_hidden_states is not None:
            self.input_buffers["previous_hidden_states"] = previous_hidden_states
        if seqlen_agnostic_capture_inputs is not None:
            self.input_buffers.update(seqlen_agnostic_capture_inputs)
        if self.is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"] = encoder_input_ids
            self.input_buffers["encoder_positions"] = encoder_positions

        self.attn_state.graph_capture_save_input_buffers(self.input_buffers)

        if isinstance(hidden_or_intermediate_states, IntermediateTensors):
            self.output_buffers.update(hidden_or_intermediate_states.items())
        else:
            self.output_buffers["hidden_states"] = hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        seqlen_agnostic_capture_inputs: Optional[Dict[str, Any]] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_positions: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Runs the model with the captured graph.

        Args:
            input_ids: The input token IDs.
            positions: The positions of the input tokens.
            kv_caches: The KV caches.
            attn_metadata: The attention metadata.
            inputs_embeds: The input embeddings. If not None, input_ids will be
                ignored.
            intermediate_tensors: The intermediate inputs from the previous pipeline
                stage.
            previous_hidden_states: The hidden states from the previous step.
                Used by draft models such as EAGLE.
            seqlen_agnostic_capture_inputs: The sequence length agnostic inputs.
                Used by Mamba-based models.
            encoder_input_ids: The input token IDs for the encoder.
            encoder_positions: The positions of the input tokens for the encoder.
        Returns:
            The output of the model. If the model is a tensor parallel model,
            the output is a single tensor. If the model is a pipeline parallel
            model, the output is an IntermediateTensors object.
        """
        # KV caches are passed in-place.
        del kv_caches
        assert self.graph is not None
        # Copy the input tensors to the input buffers.
        if inputs_embeds is not None:
            self.input_buffers["inputs_embeds"].copy_(inputs_embeds,
                                                      non_blocking=True)
        else:
            self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        if intermediate_tensors is not None:
            for k, v in intermediate_tensors.items():
                self.input_buffers[k].copy_(v, non_blocking=True)
        if previous_hidden_states is not None:
            self.input_buffers["previous_hidden_states"].copy_(
                previous_hidden_states, non_blocking=True)
        if seqlen_agnostic_capture_inputs is not None:
            for k, v in seqlen_agnostic_capture_inputs.items():
                self.input_buffers[k].copy_(v, non_blocking=True)
        if self.is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"].copy_(encoder_input_ids,
                                                          non_blocking=True)
            self.input_buffers["encoder_positions"].copy_(encoder_positions,
                                                          non_blocking=True)

        self.attn_state.graph_capture_prepare_input_buffers(
            self.input_buffers, attn_metadata)

        # Replay the graph.
        self.graph.replay()

        # Return the output buffers.
        if "hidden_states" in self.output_buffers:
            return self.output_buffers["hidden_states"]
        else:
            return IntermediateTensors(self.output_buffers)

    def __repr__(self) -> str:
        return f"CUDAGraphRunner({self.graph_name})"
