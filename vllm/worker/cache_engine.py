# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
from typing import List, Optional # Added Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        num_layers_on_this_stage: int,
        my_gpu_blocks: Optional[int] = None,
        my_cpu_blocks: Optional[int] = None,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        self.num_attention_layers = num_layers_on_this_stage
        logger.info(f"[CACHE_ENGINE_INIT_DEBUG] Initializing CacheEngine with num_attention_layers = {self.num_attention_layers} (from num_layers_on_this_stage)")
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        
        # These attributes store the configuration values from CacheConfig.
        # They might represent global totals or system-wide defaults.
        self.config_num_gpu_blocks = cache_config.num_gpu_blocks
        self.config_num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Determine the actual number of GPU blocks to allocate for this worker.
        if my_gpu_blocks is not None:
            self.worker_gpu_blocks = my_gpu_blocks
            logger.info(f"Using provided my_gpu_blocks: {self.worker_gpu_blocks} for this worker's GPU cache.")
        else:
            logger.warning(
                "my_gpu_blocks was not provided to CacheEngine. "
                "Falling back to cache_config.num_gpu_blocks (%d) for this worker's GPU cache. "
                "Ensure this is the correct value for this specific worker, especially in multi-GPU setups.",
                self.config_num_gpu_blocks
            )
            self.worker_gpu_blocks = self.config_num_gpu_blocks
        
        # Determine the actual number of CPU blocks to allocate for this worker.
        if my_cpu_blocks is not None:
            self.worker_cpu_blocks = my_cpu_blocks
            logger.info(f"Using provided my_cpu_blocks: {self.worker_cpu_blocks} for this worker's CPU cache.")
        else:
            logger.warning(
                "my_cpu_blocks was not provided to CacheEngine. "
                "Falling back to cache_config.num_cpu_blocks (%d) for this worker's CPU cache.",
                self.config_num_cpu_blocks
            )
            self.worker_cpu_blocks = self.config_num_cpu_blocks


        # Initialize the cache with worker-specific block counts
        logger.info(f"Allocating GPU KV cache with {self.worker_gpu_blocks} blocks for this worker on device {self.device_config.device_type}.")
        self.gpu_cache = self._allocate_kv_cache(
            self.worker_gpu_blocks, self.device_config.device_type)
        
        logger.info(f"Allocating CPU KV cache with {self.worker_cpu_blocks} blocks for this worker on CPU.")
        self.cpu_cache = self._allocate_kv_cache(self.worker_cpu_blocks, "cpu")
        
    # initialize_per_gpu_caches method is removed as its logic is now integrated into __init__.

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        if num_blocks == 0:
            logger.warning(
                f"Requested 0 blocks for KV cache on device '{device}'. "
                "Returning an empty list for the cache. This may be intended if no cache is needed on this device type for this worker."
            )
            return []
        
        logger.info(f"Allocating {num_blocks} blocks for KV cache on device '{device}'.")
        
        kv_cache_generic_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        try:
            kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order(
            )
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_generic_shape)))

        kv_cache_allocation_shape = tuple(kv_cache_generic_shape[i]
                                          for i in kv_cache_stride_order)

        for i in range(self.num_attention_layers):
            try:
                layer_kv_cache = torch.zeros(
                    kv_cache_allocation_shape,
                    dtype=self.dtype,
                    pin_memory=pin_memory,
                    device=device).permute(*kv_cache_stride_order)
                kv_cache.append(layer_kv_cache)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA Out of Memory during KV cache allocation for layer {i} on device '{device}' "
                             f"with shape {kv_cache_allocation_shape} (permuted from {kv_cache_generic_shape}) and dtype {self.dtype}. "
                             f"Number of blocks requested: {num_blocks}. Consider reducing block count for this GPU. Error: {e}")
                # Propagate the error to allow higher-level handling
                raise 
            except Exception as e:
                logger.error(f"An unexpected error occurred during KV cache allocation for layer {i} on device '{device}': {e}")
                raise

        logger.info(f"Successfully allocated KV cache for {self.num_attention_layers} layers on device '{device}' with {num_blocks} blocks each.")
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        if not self.gpu_cache or not self.cpu_cache:
            logger.warning("Swap_in called but GPU or CPU cache is not initialized or empty (0 blocks). Skipping operation.")
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        if not self.gpu_cache or not self.cpu_cache:
            logger.warning("Swap_out called but GPU or CPU cache is not initialized or empty (0 blocks). Skipping operation.")
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        if not self.gpu_cache:
            logger.warning("Copy called but GPU cache is not initialized or empty (0 blocks). Skipping operation.")
            return
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        # This method calculates the size of *one block*, not the total number of layers.
        # The original CacheEngine seemed to multiply by num_attention_layers here which
        # seems incorrect for a "per block" size.
        # Let's assume num_attention_layers should be 1 for a single block's memory.
        # However, the original code includes:
        # num_attention_layers = model_config.get_num_layers_by_block_type(...)
        # total = num_attention_layers * cache_config.block_size * (key_cache_entry + value_cache_entry)
        # This implies the original 'get_cache_block_size' might be calculating total memory for all layers,
        # not the size of a single "cache block" unit that num_gpu_blocks refers to.
        # Re-evaluating the original code's intent for this static method:
        # If it's the size of *one physical block in memory across all layers it would serve for one token step*, then it is:
        # (key_bytes_per_token_per_layer + value_bytes_per_token_per_layer) * num_layers
        # If it's the size of *one "page" or "slot" in the KV cache tensor (which stores multiple tokens up to block_size)*,
        # then it is:
        # (key_bytes_per_entry + value_bytes_per_entry) * block_size_in_tokens
        #
        # The original formula seems to be:
        # num_attention_layers * block_size_tokens * (bytes_per_token_key + bytes_per_token_value)
        # This looks like the memory for *all layers for a chunk of block_size tokens*.
        # This static method's purpose might be for estimating total memory for a given number of blocks,
        # where "a block" here could mean "a unit of allocation covering all layers for block_size tokens".
        # Given its usage in higher-level components (like determining num_gpu_blocks), this more holistic
        # definition of "block size" makes sense.
        # So, the original formula seems correct under that interpretation.
        
        num_layers_for_static_calc = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention
        )
        # If this CacheEngine instance is for a pipeline stage with fewer layers,
        # the static method might still be expected to calculate for the whole model.
        # This needs clarification based on how `get_cache_block_size` is used externally.
        # For now, adhering to the original implementation structure.

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry_bytes = num_heads * head_size # size for one token's key vector for one layer
        
        value_cache_entry_bytes = key_cache_entry_bytes if not model_config.use_mla else 0
        
        # Total bytes per token position for all layers this cache instance manages
        bytes_per_token_all_layers = num_layers_for_static_calc * (key_cache_entry_bytes + value_cache_entry_bytes)
        
        # Total bytes for one "block" which can hold `cache_config.block_size` tokens
        total_bytes_for_one_block = bytes_per_token_all_layers * cache_config.block_size
        
        dtype_size = get_dtype_size(dtype) # Size of the data type in bytes (e.g., float16 = 2)
        
        # The number of elements of `dtype` that fit into `total_bytes_for_one_block`
        # This seems to be what the original code computes, effectively:
        # (num_layers * block_size_tokens * (num_heads * head_size + (if not mla) num_heads * head_size)) * dtype_size
        # The multiplication by dtype_size at the end is standard.

        # Original calculation:
        # key_cache_entry = num_heads * head_size # This is num elements, not bytes
        # value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        # total_elements_one_block = num_layers_for_static_calc * cache_config.block_size * \
        #                            (key_cache_entry + value_cache_entry)
        # return dtype_size * total_elements_one_block

        # Let's stick to the original calculation structure:
        elements_per_token_key = num_heads * head_size
        elements_per_token_value = elements_per_token_key if not model_config.use_mla else 0
        
        total_elements_per_block_unit = num_layers_for_static_calc * cache_config.block_size * \
            (elements_per_token_key + elements_per_token_value)
            
        return dtype_size * total_elements_per_block_unit