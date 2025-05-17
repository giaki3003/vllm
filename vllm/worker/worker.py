# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              get_pp_group, # Corrected import
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.utils import (GiB_bytes, MemorySnapshot,
                        bind_single_layer_kv_cache,
                        memory_profiling)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.attention.backends.abstract import AttentionType
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type ==
                model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ("medusa", "mlp_speculator", "eagle", "deepseek_mtp")) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: Optional[CacheEngine] = None
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        # Reverting to List[Optional[List[Tensor]]] to match capture_model's expectation
        self.gpu_cache: Optional[List[Optional[List[torch.Tensor]]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}
        self.pipeline_stage_rank: Optional[int] = None

        # Buffers saved before sleep
        self._sleep_saved_buffers: Dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set the pipeline stage rank for this worker
        if self.parallel_config.pipeline_parallel_size > 1:
            # Ensure PP group is initialized before trying to get rank in it
            if get_pp_group() is not None:
                 self.pipeline_stage_rank = get_pp_group().rank_in_group
            else:
                # This case should ideally not happen if ensure_model_parallel_initialized was called
                logger.error("Pipeline parallel group not initialized when trying to get rank in init_device.")
                # Fallback or raise error, for now, let's assume 0 if group not ready,
                # though this might indicate a deeper issue.
                self.pipeline_stage_rank = 0
        else:
            self.pipeline_stage_rank = 0
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: Starting determine_num_available_blocks.")
        
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        :::{tip}
        You may limit the usage of GPU memory
        by adjusting the `gpu_memory_utilization` parameter.
        :::
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: Before torch.cuda.empty_cache().")
        torch.cuda.empty_cache()
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: Before torch.cuda.reset_peak_memory_stats().")
        torch.cuda.reset_peak_memory_stats()

        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: Before torch.cuda.mem_get_info().")
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: Before memory_profiling context.")
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.baseline_snapshot,
                weights_memory=self.model_runner.model_memory_usage) as result:
            self.model_runner.profile_run()

        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: After memory_profiling context. Result weights memory: {result.weights_memory}, peak: {result.torch_peak_increase}")

        self._assert_memory_footprint_increased_during_profiling()

        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance -
                                     result.non_kv_cache_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        logger.error(f"[WORKER_PROFILE_DEBUG] Worker rank {self.rank}: available_kv_cache_memory = {available_kv_cache_memory} ({available_kv_cache_memory / (1024**3):.2f} GiB), cache_block_size = {cache_block_size} ({cache_block_size / (1024**2):.2f} MiB)")
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
               "the current vLLM instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.cache_config.gpu_memory_utilization:.2f})"
               f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
               "model weights take "
               f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
               " non_torch_memory takes "
               f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
               " PyTorch activation peak memory takes "
               f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, total = torch.cuda.mem_get_info()
        cuda_memory = total - free_gpu_memory
        assert self.baseline_snapshot.cuda_memory < cuda_memory, (
            "Error in memory profiling. "
            f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
            f"currently used memory {cuda_memory}. "
            f"This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

    def initialize_cache(self, gpu_blocks_per_worker: List[int],
                         cpu_blocks_per_worker: List[int]) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        # Determine the rank of this worker to pick its specific block counts.
        # self.rank is the global rank.
        # If pipeline parallelism is used, workers are typically indexed 0 to PP_SIZE-1
        # within the context of the list passed from the executor.
        # The list gpu_blocks_per_worker should be ordered by global rank.
        my_rank_in_world = self.rank
        my_gpu_blocks = gpu_blocks_per_worker[my_rank_in_world]
        my_cpu_blocks = cpu_blocks_per_worker[my_rank_in_world]

        logger.info(f"Worker rank {my_rank_in_world}: Using GPU blocks: {my_gpu_blocks}, CPU blocks: {my_cpu_blocks}")
        
        # REMOVE THE FOLLOWING LINE:
        # self.cache_engine.initialize_per_gpu_caches(my_gpu_blocks, my_cpu_blocks)
        # This line caused the AttributeError because self.cache_engine is None at this point,
        # and the method initialize_per_gpu_caches was removed from CacheEngine.

        raise_if_cache_size_invalid(
            my_gpu_blocks, self.cache_config.block_size,
            self.cache_config.is_attention_free,
            self.model_config.max_model_len)

        # These lines are important: they ensure that when _init_cache_engine
        # creates the CacheEngine using self.cache_config, it gets the
        # correct per-worker block counts via the fallback mechanism in CacheEngine.__init__.
        self.cache_config.num_gpu_blocks = my_gpu_blocks
        self.cache_config.num_cpu_blocks = my_cpu_blocks

        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            # This method will now correctly instantiate self.cache_engine
            # using the updated self.cache_config.
            self._init_cache_engine() 
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.pipeline_stage_rank is not None, "Worker's pipeline_stage_rank not initialized"
        
        # ---- START PID and RANK IDENTIFICATION FOR LOGGING ----
        current_pid = os.getpid()
        # Target worker rank 1. PIDs can change, so using self.rank is more robust.
        is_target_worker_rank1 = (hasattr(self, 'rank') and self.rank == 1)
        log_prefix_parts = [f"pid={current_pid}"]
        if hasattr(self, 'rank'):
            log_prefix_parts.append(f"rank={self.rank}")
        if hasattr(self, 'pipeline_stage_rank'):
            log_prefix_parts.append(f"pp_stage={self.pipeline_stage_rank}")
        log_prefix = f"[WORKER_INIT_CACHE_DEBUG {' '.join(log_prefix_parts)}]"
        # ---- END PID and RANK IDENTIFICATION FOR LOGGING ----

        if self.parallel_config.pipeline_parallel_size > 1 and \
           hasattr(self.parallel_config, '_current_pipeline_stage_counts') and \
           self.parallel_config._current_pipeline_stage_counts: # Check attribute existence
            num_layers_for_this_stage = self.parallel_config._current_pipeline_stage_counts[self.pipeline_stage_rank]
            if num_layers_for_this_stage == 0: 
                logger.warning(
                    f"{log_prefix} Stage has 0 layers assigned based on VRAM balancing. "
                    f"CacheEngine will be initialized for 1 layer to avoid errors, but this stage might be unused.")
                num_layers_for_this_stage = 1 
        else:
            num_layers_for_this_stage = self.model_config.get_num_layers(self.parallel_config)
            if num_layers_for_this_stage == 0: 
                logger.warning(
                    f"{log_prefix} num_layers_for_this_stage calculated as 0. "
                    f"Defaulting to 1 layer for CacheEngine. Total model layers: {self.model_config.get_num_layers(self.parallel_config)}")
                num_layers_for_this_stage = 1

        logger.info(f"{log_prefix} Initializing CacheEngine for {num_layers_for_this_stage} layers. "
                    f"Using cache_config.num_gpu_blocks: {self.cache_config.num_gpu_blocks}, "
                    f"cache_config.num_cpu_blocks: {self.cache_config.num_cpu_blocks}")
        assert self.cache_config.num_gpu_blocks is not None, \
            f"{log_prefix} num_gpu_blocks is None before CacheEngine init."

        self.cache_engine = CacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            self.device_config,
            num_layers_for_this_stage
        )
        
        # -------- START LOGGING single_engine_gpu_cache --------
        single_engine_gpu_cache = self.cache_engine.gpu_cache 

        if is_target_worker_rank1: 
            logger.error(f"{log_prefix} For Target Worker (Rank 1):") # Use logger.error for high visibility
            logger.error(f"{log_prefix}   single_engine_gpu_cache (from self.cache_engine.gpu_cache): type={type(single_engine_gpu_cache)}")
            if isinstance(single_engine_gpu_cache, list):
                logger.error(f"{log_prefix}   single_engine_gpu_cache length: {len(single_engine_gpu_cache)} (expected {num_layers_for_this_stage} for this worker)")
                if not single_engine_gpu_cache: 
                     logger.error(f"{log_prefix}   CRITICAL: single_engine_gpu_cache is an EMPTY LIST.")
                for i, tensor_cache in enumerate(single_engine_gpu_cache):
                    if tensor_cache is not None:
                        logger.error(
                            f"{log_prefix}   Layer {i} tensor_cache - "
                            f"Shape: {tensor_cache.shape}, Numel: {tensor_cache.numel()}, "
                            f"Dtype: {tensor_cache.dtype}, Device: {tensor_cache.device}"
                        )
                        if tensor_cache.numel() == 0:
                            logger.error(f"{log_prefix}   CRITICAL: Layer {i} tensor_cache has Numel == 0.")
                    else:
                        logger.error(f"{log_prefix}   CRITICAL: Layer {i} tensor_cache is None.")
            elif single_engine_gpu_cache is None:
                 logger.error(f"{log_prefix}   CRITICAL: single_engine_gpu_cache is None.")
            else:
                logger.error(f"{log_prefix}   single_engine_gpu_cache is NOT a list, type: {type(single_engine_gpu_cache)}.")
        # -------- END LOGGING single_engine_gpu_cache --------

        self.gpu_cache = [None] * self.parallel_config.pipeline_parallel_size
        
        # Check single_engine_gpu_cache before assignment
        if single_engine_gpu_cache is not None and (not isinstance(single_engine_gpu_cache, list) or len(single_engine_gpu_cache) > 0):
            # Condition updated: ensure it's not None AND (if it's a list, it's not empty)
            assert self.pipeline_stage_rank is not None, f"{log_prefix} pipeline_stage_rank is None during gpu_cache setup"
            self.gpu_cache[self.pipeline_stage_rank] = single_engine_gpu_cache
            if is_target_worker_rank1:
                 logger.error(f"{log_prefix} Target Worker (Rank 1): Assigned single_engine_gpu_cache (type: {type(single_engine_gpu_cache)}, len: {len(single_engine_gpu_cache) if isinstance(single_engine_gpu_cache, list) else 'N/A'}) to self.gpu_cache[{self.pipeline_stage_rank}]")
        else:
            # This 'else' branch will be hit if single_engine_gpu_cache is None or an empty list.
            # self.gpu_cache[self.pipeline_stage_rank] will remain None.
            log_level_for_issue = logger.error if is_target_worker_rank1 else logger.warning
            log_level_for_issue(
                f"{log_prefix} "
                f"{'CRITICAL' if is_target_worker_rank1 else 'WARNING'}: "
                f"single_engine_gpu_cache is None or empty (type: {type(single_engine_gpu_cache)}, "
                f"len={len(single_engine_gpu_cache) if isinstance(single_engine_gpu_cache, list) else 'N/A'}) "
                f"after CacheEngine init. "
                f"self.gpu_cache[{self.pipeline_stage_rank}] will remain None."
            )

        # KV cache binding logic (remains the same)
        static_forward_root_ctx = self.compilation_config.static_forward_context
        assert self.pipeline_stage_rank is not None, \
            f"{log_prefix} pipeline_stage_rank is None in _init_cache_engine for binding."
        pp_rank = self.pipeline_stage_rank
        pp_size = self.parallel_config.pipeline_parallel_size
        
        local_layer_idx = 0
        # Re-fetch single_engine_gpu_cache for binding, as it's the List[Tensor] for this stage
        # This ensures we use what was actually put into self.gpu_cache[pp_rank] if anything.
        # Or, more directly, use self.gpu_cache[pp_rank] if it's guaranteed to be List[Tensor] here
        cache_to_bind = self.gpu_cache[pp_rank] # This is what ModelRunner will eventually use parts of.

        if cache_to_bind is not None and isinstance(cache_to_bind, list) and \
           hasattr(self.model_runner, 'model') and self.model_runner.model is not None:
            if is_target_worker_rank1:
                logger.error(f"{log_prefix} Target Worker (Rank 1): Proceeding with KV cache binding. cache_to_bind length: {len(cache_to_bind)}")
            for global_name, module in self.model_runner.model.named_modules():
                is_cacheable_layer = hasattr(module, 'attn_type') and \
                                     module.attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)
                
                if is_cacheable_layer and not isinstance(module, PPMissingLayer):
                    if global_name in static_forward_root_ctx:
                        layer_static_ctx = static_forward_root_ctx[global_name]
                        if local_layer_idx < len(cache_to_bind):
                            cache_tensor_for_this_layer = cache_to_bind[local_layer_idx]
                            if cache_tensor_for_this_layer is None or cache_tensor_for_this_layer.numel() == 0 :
                                if is_target_worker_rank1:
                                    logger.error(f"{log_prefix} Target Worker (Rank 1): For binding layer {global_name} (local_idx {local_layer_idx}), "
                                                 f"cache_tensor_for_this_layer is None or empty! Shape: {cache_tensor_for_this_layer.shape if cache_tensor_for_this_layer is not None else 'None'}")
                                # Depending on strictness, you might want to skip binding or error out.
                                # For now, original logic would proceed and potentially fail in bind_single_layer_kv_cache if tensor is bad.
                            
                            bind_single_layer_kv_cache(
                                layer_static_ctx,
                                cache_tensor_for_this_layer,
                                pp_rank,
                                pp_size
                            )
                            local_layer_idx += 1
                        else:
                            logger.error(f"{log_prefix} "
                                       f"Mismatch between iterable model layers and cache_to_bind size. "
                                       f"Attempted to access cache_to_bind index {local_layer_idx} "
                                       f"for layer {global_name}, but cache size is {len(cache_to_bind)}.")
                            break 
            
            if cache_to_bind and local_layer_idx != len(cache_to_bind):
                 logger.warning(f"{log_prefix} Not all tensors in cache_to_bind were bound to layers. Bound: {local_layer_idx}, Total: {len(cache_to_bind)}")
                 
        elif cache_to_bind is None or not isinstance(cache_to_bind, list):
            logger.warning(f"{log_prefix} cache_to_bind is None or not a list (type: {type(cache_to_bind)}), skipping kv_cache binding.")
        elif not hasattr(self.model_runner, 'model') or self.model_runner.model is None:
            logger.warning(f"{log_prefix} model_runner.model not available, skipping kv_cache binding.")

    def _warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        request_target_stage = worker_input.virtual_engine
        
        # Worker should only perform cache operations if the request is for its stage
        if request_target_stage == self.pipeline_stage_rank:
            assert self.cache_engine is not None, \
                f"Worker rank {self.rank} (PP stage {self.pipeline_stage_rank}): cache_engine is None in execute_worker."
            # Issue cache operations.
            if (worker_input.blocks_to_swap_in is not None
                    and worker_input.blocks_to_swap_in.numel() > 0):
                self.cache_engine.swap_in(
                    worker_input.blocks_to_swap_in)
            if (worker_input.blocks_to_swap_out is not None
                    and worker_input.blocks_to_swap_out.numel() > 0):
                self.cache_engine.swap_out(
                    worker_input.blocks_to_swap_out)
            if (worker_input.blocks_to_copy is not None
                    and worker_input.blocks_to_copy.numel() > 0):
                self.cache_engine.copy(worker_input.blocks_to_copy)
        # else:
            # Optionally, log if a worker for one stage receives a cache op request for another.
            # This might indicate an issue upstream or be expected if requests are broadcast.
            # logger.debug(f"Worker rank {self.rank} (PP stage {self.pipeline_stage_rank}) "
            #              f"received cache op for stage {request_target_stage}, ignoring.")

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                                SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta)
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[
                        request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id])

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,
                                max_model_len) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError("No memory should be allocated for the cache blocks "
                         f"for an attention-free model, but {num_gpu_blocks} "
                         "blocks are allocated.")
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
