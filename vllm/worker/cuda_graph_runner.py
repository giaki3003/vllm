# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionState
# Assuming IntermediateTensors might be used or returned by the model within the graph
from vllm.sequence import IntermediateTensors 

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata


class CUDAGraphRunner(nn.Module):

    def __init__(self, model: nn.Module, backend_name: str,
                 attn_state_clone: AttentionState,
                 is_encoder_decoder_model: bool):
        super().__init__()
        self.model = model
        self.backend_name = backend_name
        self.attn_state_clone = attn_state_clone
        self.is_encoder_decoder_model = is_encoder_decoder_model
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.static_input_names: List[str] = []
        self.static_output: Optional[torch.Tensor | IntermediateTensors | Tuple[torch.Tensor, ...]] = None


    @property
    def graph_capture_inputs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(self.input_buffers[name]
                     for name in self.static_input_names)

    @property
    def graph_capture_outputs(self) -> Tuple[torch.Tensor, ...]:
        # If the model returns a tuple, we need to flatten it.
        if isinstance(self.static_output, tuple):
            return tuple(self.static_output)
        elif isinstance(self.static_output, IntermediateTensors):
            # If IntermediateTensors, its tensors attribute should be a tuple or dict
            # For simplicity, assuming it can be handled or this part needs adjustment
            # based on how IntermediateTensors are structured for graph output.
            # This might need a more specific handling if IntermediateTensors are complex.
            if isinstance(self.static_output.tensors, tuple):
                return self.static_output.tensors
            elif isinstance(self.static_output.tensors, dict):
                return tuple(self.static_output.tensors.values())
            elif isinstance(self.static_output.tensors, torch.Tensor): # Handle single tensor case
                return (self.static_output.tensors,)
            else: 
                return tuple()
        elif self.static_output is None:
            return tuple()
        return (self.static_output, )


    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        memory_pool: Optional[Tuple[int, int]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        seqlen_agnostic_capture_inputs: Optional[Dict[str, Any]] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_positions: Optional[torch.Tensor] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        assert self.graph is None
        self.static_input_names = []

        if inputs_embeds is not None:
            self.input_buffers["inputs_embeds"] = inputs_embeds
            self.static_input_names.append("inputs_embeds")
        else:
            self.input_buffers["input_ids"] = input_ids
            self.static_input_names.append("input_ids")
        self.input_buffers["positions"] = positions
        self.static_input_names.append("positions")

        if previous_hidden_states is not None:
            self.input_buffers["previous_hidden_states"] = previous_hidden_states
            self.static_input_names.append("previous_hidden_states")

        if intermediate_tensors is not None:
            if isinstance(intermediate_tensors, torch.Tensor): # Check if it's a tensor
                 self.input_buffers["intermediate_tensors"] = intermediate_tensors
                 self.static_input_names.append("intermediate_tensors")
            elif isinstance(intermediate_tensors, IntermediateTensors) and \
                 isinstance(intermediate_tensors.tensors, torch.Tensor): # Check if .tensors is a tensor
                 self.input_buffers["intermediate_tensors"] = intermediate_tensors.tensors
                 self.static_input_names.append("intermediate_tensors")


        if self.is_encoder_decoder_model:
            assert encoder_input_ids is not None
            assert encoder_positions is not None
            self.input_buffers["encoder_input_ids"] = encoder_input_ids
            self.input_buffers["encoder_positions"] = encoder_positions
            self.static_input_names.extend(
                ["encoder_input_ids", "encoder_positions"])

        if seqlen_agnostic_capture_inputs:
            for name, buf in seqlen_agnostic_capture_inputs.items():
                if isinstance(buf, torch.Tensor): # Ensure buffer is a tensor
                    self.input_buffers[name] = buf
                    self.static_input_names.append(name)

        model_inputs_for_warmup = {
            name: buf
            for name, buf in self.input_buffers.items()
            if name in self.static_input_names 
        }
        
        output_hidden_or_intermediate_states = self.model(
            **model_inputs_for_warmup
            # kv_caches and attn_metadata are not direct arguments to the model's
            # top-level forward method. They are used by the Attention layers
            # internally, accessed via attn_metadata (from ForwardContext) and
            # the CacheEngine.
        )

        if isinstance(output_hidden_or_intermediate_states,
                      IntermediateTensors):
            self.static_output = output_hidden_or_intermediate_states
            if isinstance(output_hidden_or_intermediate_states.tensors, torch.Tensor):
                self.output_buffers["intermediate_tensors_output"] = output_hidden_or_intermediate_states.tensors.clone()
            # Handle dict/tuple if necessary for output_buffers
        elif isinstance(output_hidden_or_intermediate_states, torch.Tensor):
            self.static_output = output_hidden_or_intermediate_states
            self.output_buffers["hidden_states"] = self.static_output.clone()
        elif isinstance(output_hidden_or_intermediate_states, tuple):
            self.static_output = output_hidden_or_intermediate_states
            for i, out_tensor in enumerate(output_hidden_or_intermediate_states):
                if isinstance(out_tensor, torch.Tensor):
                    self.output_buffers[f"output_{i}"] = out_tensor.clone()


        if stream is None:
            stream = torch.cuda.current_stream()

        self.graph = torch.cuda.CUDAGraph()
        memory_pool_kwargs = dict(pool=memory_pool) if memory_pool is not None else {}

        with torch.cuda.graph(self.graph, stream=stream, **memory_pool_kwargs):
            graph_run_inputs = {name: self.input_buffers[name] for name in self.static_input_names}
            
            current_output = self.model(
                **graph_run_inputs
                # kv_caches and attn_metadata are not direct arguments here either.
            )
            if isinstance(current_output, IntermediateTensors):
                if isinstance(current_output.tensors, torch.Tensor) and "intermediate_tensors_output" in self.output_buffers:
                     self.output_buffers["intermediate_tensors_output"].copy_(current_output.tensors)
            elif isinstance(current_output, torch.Tensor) and "hidden_states" in self.output_buffers:
                self.output_buffers["hidden_states"].copy_(current_output)
            elif isinstance(current_output, tuple):
                 for i, out_tensor in enumerate(current_output):
                     if isinstance(out_tensor, torch.Tensor) and f"output_{i}" in self.output_buffers:
                         self.output_buffers[f"output_{i}"].copy_(out_tensor)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        finished_requests_ids: Optional[List[str]] = None, # Added based on original model_runner call
        request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None, # Added
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_positions: Optional[torch.Tensor] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor] | IntermediateTensors: 
        assert self.graph is not None
        if inputs_embeds is not None and "inputs_embeds" in self.input_buffers:
            self.input_buffers["inputs_embeds"].copy_(inputs_embeds, non_blocking=True)
        elif "input_ids" in self.input_buffers : # Ensure key exists before copying
            self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        
        if "positions" in self.input_buffers: # Ensure key exists
            self.input_buffers["positions"].copy_(positions, non_blocking=True)

        if previous_hidden_states is not None and "previous_hidden_states" in self.input_buffers:
            self.input_buffers["previous_hidden_states"].copy_(previous_hidden_states, non_blocking=True)
        
        if intermediate_tensors is not None and "intermediate_tensors" in self.input_buffers:
            tensor_to_copy = None
            if isinstance(intermediate_tensors, torch.Tensor):
                tensor_to_copy = intermediate_tensors
            elif isinstance(intermediate_tensors, IntermediateTensors) and \
                 isinstance(intermediate_tensors.tensors, torch.Tensor):
                tensor_to_copy = intermediate_tensors.tensors
            
            if tensor_to_copy is not None and isinstance(self.input_buffers["intermediate_tensors"], torch.Tensor):
                self.input_buffers["intermediate_tensors"].copy_(tensor_to_copy, non_blocking=True)


        if self.is_encoder_decoder_model:
            assert encoder_input_ids is not None
            assert encoder_positions is not None
            if "encoder_input_ids" in self.input_buffers: # Ensure key exists
                self.input_buffers["encoder_input_ids"].copy_(encoder_input_ids, non_blocking=True)
            if "encoder_positions" in self.input_buffers: # Ensure key exists
                self.input_buffers["encoder_positions"].copy_(encoder_positions, non_blocking=True)

        self.attn_state_clone.set_attn_metadata(attn_metadata)
        self.attn_state_clone.set_kv_caches(kv_caches)

        self.graph.replay()

        if isinstance(self.static_output, IntermediateTensors):
            return self.static_output 
        
        return self.output_buffers.copy() # Return a copy to avoid external modification