# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import cloudpickle
import msgspec

import vllm.envs as envs
from vllm.executor.executor_base import (
    DistributedExecutorBase)  # yapf: disable
from vllm.executor.msgspec_utils import encode_hook
from vllm.executor.ray_utils import (RayWorkerWrapper, initialize_ray_cluster,
                                     ray)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)

from collections import Counter

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


@dataclass
class RayWorkerMetaData:
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """
    worker: ActorHandle
    created_rank: int
    adjusted_rank: int = -1
    ip: str = ""


class RayDistributedExecutor(DistributedExecutorBase):
    """Ray-based distributed executor"""

    # These env vars are worker-specific, therefore are NOT copied
    # from the driver to the workers
    WORKER_SPECIFIC_ENV_VARS = {
        "VLLM_HOST_IP", "VLLM_HOST_PORT", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"
    }

    config_home = envs.VLLM_CONFIG_ROOT
    # This file contains a list of env vars that should not be copied
    # from the driver to the Ray workers.
    non_carry_over_env_vars_file = os.path.join(
        config_home, "ray_non_carry_over_env_vars.json")
    if os.path.exists(non_carry_over_env_vars_file):
        with open(non_carry_over_env_vars_file) as f:
            non_carry_over_env_vars = set(json.load(f))
    else:
        non_carry_over_env_vars = set()

    uses_ray: bool = True

    def _init_executor(self) -> None:
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None
        if envs.VLLM_USE_V1:
            # V1 uses SPMD worker and compiled DAG
            os.environ["VLLM_USE_RAY_SPMD_WORKER"] = "1"
            os.environ["VLLM_USE_RAY_COMPILED_DAG"] = "1"

            # For TPU, avoid compiling NVIDIA's NCCL
            if current_platform.is_tpu():
                os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"

        # If the env var is set, it uses the Ray's compiled DAG API
        # which optimizes the control plane overhead.
        # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
        # Currently, this requires USE_RAY_SPMD_WORKER=True.
        self.use_ray_compiled_dag = envs.VLLM_USE_RAY_COMPILED_DAG
        # If the env var is set, then we do not distinguish between the
        # "driver worker" vs other workers. Also, the rank 0 worker will
        # be executed in a remote Ray worker. Currently this requires
        # USE_RAY_COMPILED_DAG=True.
        self.use_ray_spmd_worker = envs.VLLM_USE_RAY_SPMD_WORKER
        if self.use_ray_compiled_dag:
            assert self.use_ray_spmd_worker, (
                "VLLM_USE_RAY_COMPILED_DAG=1 requires "
                "VLLM_USE_RAY_SPMD_WORKER=1")
        if self.use_ray_spmd_worker:
            # TODO: Support SPMD worker for non-DAG Ray executor.
            assert self.use_ray_compiled_dag, (
                "VLLM_USE_RAY_SPMD_WORKER=1 requires "
                "VLLM_USE_RAY_COMPILED_DAG=1")

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        self.input_encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)
        self.output_decoder = msgspec.msgpack.Decoder(
            Optional[List[SamplerOutput]])
        self.use_v1 = envs.VLLM_USE_V1

        self.pp_locks: Optional[List[asyncio.Lock]] = None
        if not self.use_ray_compiled_dag:
            self.driver_exec_method = make_async(
                self.driver_worker.execute_method)

    def shutdown(self) -> None:
        logger.info(
            "Shutting down Ray distributed executor. If you see error log "
            "from logging.cc regarding SIGTERM received, please ignore because "
            "this is the expected termination process in Ray.")
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    def _configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas",
                "o": "'worker_process_%p'",
                "cuda-graph-trace": "node",
            }
        })

        return ray_remote_kwargs

    # child class could overwrite this to return actual env vars.
    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):

        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: Optional[RayWorkerWrapper] = None
        # The remaining workers are the actual ray actors.
        self.workers: List[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)


        # Create the workers.
        bundle_indices: List[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            try:
                bundle_indices = list(
                    map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
                assert len(bundle_indices) == self.parallel_config.world_size, \
                ("VLLM_RAY_BUNDLE_INDICES must have the same size"
                 f" as the world size, but got {bundle_indices=} "
                 f"and {self.parallel_config.world_size=}")
                assert len(set(bundle_indices)) == len(bundle_indices), \
                ("VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                 f" but got {bundle_indices=}")
                logger.info(f"[DEBUG RAY INIT] Parsed bundle_indices: {bundle_indices}")
            except Exception as e:
                logger.error(f"[DEBUG RAY INIT] Failed to parse VLLM_RAY_BUNDLE_INDICES: {e}")
                raise
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            if placement_group and placement_group.bundle_specs:
                for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                    # Use current_platform.ray_device_key which should be "GPU" for CUDA
                    if bundle.get(current_platform.ray_device_key, 0):
                        bundle_indices.append(bundle_id)

                if len(bundle_indices) < self.parallel_config.world_size:
                bundle_indices = bundle_indices[:self.parallel_config.world_size]
            else:
                bundle_indices = []

        worker_metadata: List[RayWorkerMetaData] = []
        try:
            # Assuming get_ip() is defined in ray_utils or globally accessible
            driver_ip = get_ip()
        except Exception as e:
            driver_ip = "?.?.?.?" # Assign a dummy value

        if 'bundle_indices' not in locals():
            bundle_indices = []

        for rank, bundle_id in enumerate(bundle_indices):
            worker = None
            try:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )

                device_key = current_platform.ray_device_key

                remote_args = dict(
                    num_cpus=0,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )

                if device_key == "GPU":
                    remote_args["num_gpus"] = num_gpus
                else:
                    remote_args["resources"] = {device_key: num_gpus}

                # Assuming RayWorkerWrapper is imported correctly
                worker = ray.remote(**remote_args)(RayWorkerWrapper).remote(
                    vllm_config=self.vllm_config, rpc_rank=rank
                )

            except Exception as e:
                logger.error(f"[DEBUG RAY INIT] Loop rank {rank}: FAILED to create Ray actor!")
                logger.error(traceback.format_exc())
                pass

            if worker is None:
                logger.error(f"[DEBUG RAY INIT] Loop rank {rank}: 'worker' is None after creation attempt. Skipping metadata.")
                raise RuntimeError(f"Failed to create Ray actor for rank {rank}. Check logs above.")
            else:
                logger.info(f"[DEBUG RAY INIT] Loop rank {rank}: Appending metadata for worker: {worker}")
                worker_metadata.append(
                    RayWorkerMetaData(worker=worker, created_rank=rank))


        if not worker_metadata or len(worker_metadata) < self.parallel_config.world_size:
             logger.error(f"[DEBUG RAY INIT] Failed to create all required workers. "
                          f"Expected {self.parallel_config.world_size}, got {len(worker_metadata)}. Aborting.")
             raise RuntimeError("Failed to create all required Ray workers.")

        try:
            worker_ips = ray.get([
                each.worker.get_node_ip.remote() # type: ignore[attr-defined]
                for each in worker_metadata
            ], timeout=60.0)
        except Exception as e:
            logger.error(f"[DEBUG RAY INIT] Failed to get IPs from Ray workers: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to get IPs from Ray workers. Check logs and Ray cluster status. Error: {e}")

        for i, ip in enumerate(worker_ips):
            if i < len(worker_metadata):
                worker_metadata[i].ip = ip if ip is not None else "IP_QUERY_FAILED"
            else:
                 logger.warning(f"[DEBUG RAY INIT] Index {i} out of bounds for worker_metadata while assigning IPs.")

        # --- FIX for ip_counts NameError ---
        # Calculate ip_counts based on the reported IPs in worker_metadata
        valid_ips = [md.ip for md in worker_metadata if md.ip and md.ip != "IP_QUERY_FAILED"]
        # Also consider the driver_ip itself if it's not None/dummy
        if driver_ip != "?.?.?.?":
             valid_ips.append(driver_ip) # Add driver ip to the list for counting consistency
        ip_counts = Counter(valid_ips)
        # --- END ip_counts FIX ---

        # === MODIFIED LOGIC FOR FINDING DRIVER DUMMY WORKER ===
        if not self.use_ray_spmd_worker:
            driver_dummy_worker_found_by_rank = False
            driver_worker_index = -1
            target_driver_rank = 0

            for i, md in enumerate(worker_metadata):
                if md.created_rank == target_driver_rank:
                    self.driver_dummy_worker = md.worker
                    # Assuming RayWorkerWrapper import is correct
                    self.driver_worker = RayWorkerWrapper(
                        vllm_config=self.vllm_config, rpc_rank=target_driver_rank)
                    driver_worker_index = i
                    driver_dummy_worker_found_by_rank = True
                    break

            if driver_worker_index != -1:
                 logger.info(f"[DEBUG RAY INIT] Removing worker {driver_worker_index} (rank {target_driver_rank}) from worker_metadata list as it's the driver dummy.")
                 if driver_worker_index < len(worker_metadata):
                     worker_metadata.pop(driver_worker_index)
                 else:
                      logger.error(f"[DEBUG RAY INIT] Invalid index {driver_worker_index} for popping from worker_metadata (size {len(worker_metadata)})")
            elif not driver_dummy_worker_found_by_rank:
                 logger.error(f"[DEBUG RAY INIT] CRITICAL: Could not find worker with rank {target_driver_rank} to assign as driver dummy!")

        # === END OF MODIFIED LOGIC ===

        # --- Define the sorting key function (moved here for clarity) ---
        def sort_by_driver_then_worker_ip(driver_ip_val, ip_counts_val):
            def sort_key(worker_meta_data: RayWorkerMetaData):
                ip = worker_meta_data.ip
                if ip is None or ip == "IP_QUERY_FAILED":
                    return (2, 0, "") # Put failed IPs last
                # Sort driver node first (0), then by IP frequency (ascending), then by IP string
                return (0 if ip == driver_ip_val else 1, ip_counts_val.get(ip, 0), ip)
            return sort_key

        # Now use the key function factory when sorting
        logger.info("[DEBUG RAY INIT] Sorting worker_metadata...")
        try:
            # Pass the calculated ip_counts and driver_ip to the key function factory
            sorted_worker_metadata = sorted(
                 worker_metadata,
                 key=sort_by_driver_then_worker_ip(driver_ip, ip_counts))
            logger.info(f"[DEBUG RAY INIT] Sorted worker_metadata (by rank): {[md.created_rank for md in sorted_worker_metadata]}")
            # Assign self.workers based on the sorted list
            self.workers = [each.worker for each in sorted_worker_metadata]
        except Exception as e:
             logger.error(f"[DEBUG RAY INIT] Failed during sorting: {e}")
             logger.error(traceback.format_exc())
             # Fallback: Use unsorted list if sorting fails?
             logger.warning("[DEBUG RAY INIT] Sorting failed, using unsorted worker list for self.workers.")
             self.workers = [each.worker for each in worker_metadata]


        logger.info(f"[DEBUG RAY INIT] Final self.workers list (size {len(self.workers)}): {self.workers}")
        logger.info(f"[DEBUG RAY INIT] Final self.driver_dummy_worker: {self.driver_dummy_worker}")
        logger.info(f"[DEBUG RAY INIT] Final self.driver_worker (created if dummy found): {getattr(self, 'driver_worker', 'Not Created')}")

        # This check should now pass if rank 0 worker was successfully found and assigned
        if not self.use_ray_spmd_worker and self.driver_dummy_worker is None:
            logger.error(f"[DEBUG RAY INIT] Raising ValueError: driver_dummy_worker is None (Rank {target_driver_rank} not found?). "
                         f"Driver IP={driver_ip}, Detected Worker IPs={worker_ips}")
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node (Failed to identify driver worker by rank)."
                f"Driver IP: {driver_ip}, worker IPs: {worker_ips}."
                "Check Ray scheduling and worker initialization for rank 0.")


        logger.info("[DEBUG RAY INIT] Successfully initialized Ray workers and identified driver worker.")

        # Assign workers to the pp_tp_workers grid
        logger.info("[DEBUG RAY INIT] Assigning workers to pp_tp_workers grid...")
        self.pp_tp_workers = [
            [None] * self.parallel_config.tensor_parallel_size
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Reconstruct the full worker map including the identified driver dummy
        worker_map_by_rank = {}
        # Use the *sorted* list for self.workers now
        for md in sorted_worker_metadata: # Ranks != target_driver_rank
            worker_map_by_rank[md.created_rank] = md.worker
        if self.driver_dummy_worker:
            # Add the dummy worker back using its assumed rank
            worker_map_by_rank[target_driver_rank] = self.driver_dummy_worker

        expected_worker_count = self.parallel_config.world_size
        if len(worker_map_by_rank) != expected_worker_count:
             logger.warning(f"[DEBUG RAY INIT] Mismatch between number of workers in map "
                            f"({len(worker_map_by_rank)}) and world size "
                            f"({expected_worker_count}) before assigning to grid.")

        assigned_ranks = set()
        logger.info(f"[DEBUG RAY INIT] Using direct calculation for PP/TP ranks based on TP size: {self.parallel_config.tensor_parallel_size}")
        for rank in range(expected_worker_count):
            # === MODIFIED RANK CALCULATION ===
            tp_size = self.parallel_config.tensor_parallel_size
            # Ensure tp_size is not zero to avoid division error
            if tp_size == 0:
                 logger.error("[DEBUG RAY INIT] Tensor parallel size is zero, cannot calculate ranks!")
                 raise ValueError("Tensor parallel size cannot be zero.")
            pp_rank = rank // tp_size
            tp_rank = rank % tp_size
            logger.debug(f"[DEBUG RAY INIT] Calculated for rank {rank}: pp_rank={pp_rank}, tp_rank={tp_rank}")
            # === END MODIFIED RANK CALCULATION ===

            if rank in worker_map_by_rank:
                 # Check if indices are valid before assignment
                 if pp_rank < len(self.pp_tp_workers) and tp_rank < len(self.pp_tp_workers[pp_rank]):
                     self.pp_tp_workers[pp_rank][tp_rank] = worker_map_by_rank[rank]
                     assigned_ranks.add(rank)
                     logger.info(f"[DEBUG RAY INIT] Assigned worker rank {rank} (from map) to pp_tp_workers[{pp_rank}][{tp_rank}]")
                 else:
                      logger.error(f"[DEBUG RAY INIT] Calculated indices out of bounds! rank={rank}, pp_rank={pp_rank}, tp_rank={tp_rank}, grid shape=({len(self.pp_tp_workers)} x {len(self.pp_tp_workers[0]) if self.pp_tp_workers and len(self.pp_tp_workers[0]) > tp_rank else 'invalid'})")
                      raise IndexError(f"Calculated grid indices out of bounds for rank {rank}")
            else:
                 logger.error(f"[DEBUG RAY INIT] Could not find worker for rank {rank} in worker_map_by_rank!")
                 raise RuntimeError(f"Internal error: Worker for rank {rank} not found during grid assignment.")


        if len(assigned_ranks) != expected_worker_count:
             logger.error(f"[DEBUG RAY INIT] Failed to assign all ranks to the grid! Assigned: {assigned_ranks}, Expected: {set(range(expected_worker_count))}")
             # Decide how to handle: maybe raise an error?
             raise RuntimeError("Failed to assign all ranks to the grid.")

        logger.info(f"[DEBUG RAY INIT] pp_tp_workers grid assignment complete. Example worker at [0][0]: {self.pp_tp_workers[0][0]}")
        logger.info("[DEBUG RAY INIT] Exiting _init_workers_ray successfully.")

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(worker_metadata,
                                        key=sort_by_driver_then_worker_ip)
        start_rank = 0 if self.use_ray_spmd_worker else 1
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i + start_rank
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank
            for item in sorted_worker_metadata
        }
        self._run_workers("adjust_rank", rerank_mapping)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote()) \
            ) # type: ignore

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        #if n_nodes != n_ips:
            #raise RuntimeError(
            #    f"Every node should have a unique IP address. Got {n_nodes}"
            #    f" nodes with node ids {list(node_workers.keys())} and "
            #    f"{n_ips} unique IP addresses {all_ips}. Please check your"
            #    " network configuration. If you set `VLLM_HOST_IP`"
            #    " environment variable, make sure it is unique for"
            #    " each node.")

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [{
            current_platform.device_control_env_var:
            ",".join(map(str, node_gpus[node_id])),
        } for (node_id, _) in worker_node_and_gpu_ids]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = [
            v for v in envs.environment_variables
            if v not in self.WORKER_SPECIFIC_ENV_VARS
            and v not in self.non_carry_over_env_vars
        ]

        env_vars_to_copy.extend(current_platform.additional_env_vars)

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        logger.info("non_carry_over_env_vars from config: %s",
                    self.non_carry_over_env_vars)
        logger.info(
            "Copying the following environment variables to workers: %s",
            [v for v in env_vars_to_copy if v in os.environ])
        logger.info(
            "If certain env vars should NOT be copied to workers, add them to "
            "%s file", self.non_carry_over_env_vars_file)

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          self._get_env_vars_to_be_updated())

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self._run_workers("init_worker", all_kwargs)

        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

        if self.use_ray_spmd_worker:
            for pp_rank in range(self.parallel_config.pipeline_parallel_size):
                self.pp_tp_workers.append([])
                for tp_rank in range(
                        self.parallel_config.tensor_parallel_size):
                    # PP=2, TP=4
                    # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                    rank = (pp_rank * self.parallel_config.tensor_parallel_size
                            ) + tp_rank
                    assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                    assert pp_rank < len(self.pp_tp_workers)
                    self.pp_tp_workers[pp_rank].append(self.workers[rank])

        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []

        # Enforce rank order for correct rank to return final output.
        for index, worker in enumerate(self.workers):
            # The driver worker is rank 0 and not in self.workers.
            rank = index + 1
            if rank % self.parallel_config.tensor_parallel_size == 0:
                self.tp_driver_workers.append(worker)
            else:
                self.non_driver_workers.append(worker)

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        assert not self.use_ray_spmd_worker, (
            "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1")
        return self.driver_worker.execute_method("execute_model",
                                                 execute_model_req)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if not self.use_ray_spmd_worker:
            return super().execute_model(execute_model_req)

        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        if self.use_v1:
            serialized_data = execute_model_req
        else:
            serialized_data = self.input_encoder.encode(execute_model_req)
        outputs = ray.get(self.forward_dag.execute(serialized_data))
        if self.use_v1:
            output = outputs[0]
        else:
            output = self.output_decoder.decode(outputs[0])
        return output

    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        """
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method
        if self.use_ray_spmd_worker:
            assert not async_run_tensor_parallel_workers_only, (
                "async_run_tensor_parallel_workers_only is not supported for "
                "spmd mode.")

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the ray workers first.
        ray_workers = self.workers
        if async_run_tensor_parallel_workers_only:
            ray_workers = self.non_driver_workers
        ray_worker_outputs = [
            worker.execute_method.remote(sent_method, *args, **kwargs)
            for worker in ray_workers
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        driver_worker_output = []
        # In SPMD mode, the driver worker is the same as any other worker,
        # so we only explicitly execute on the driver worker if using a
        # non-SPMD worker class.
        if not self.use_ray_spmd_worker:
            # Start the driver worker after all the ray workers.
            driver_worker_output = [
                self.driver_worker.execute_method(sent_method, *args, **kwargs)
            ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return driver_worker_output + ray_worker_outputs

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        ray.get(parallel_worker_tasks)

    def _check_ray_cgraph_installation(self):
        import pkg_resources
        from packaging import version

        required_version = version.parse("2.43.0")
        current_version = version.parse(
            pkg_resources.get_distribution("ray").version)
        if current_version < required_version:
            raise ValueError(f"Ray version {required_version} is "
                             f"required, but found {current_version}")

        import importlib.util
        cgraph_spec = importlib.util.find_spec(
            "ray.experimental.compiled_dag_ref")
        if cgraph_spec is None:
            raise ValueError("Ray Compiled Graph is not installed. "
                             "Run `pip install ray[cgraph]` to install it.")

        cupy_spec = importlib.util.find_spec("cupy")
        if (cupy_spec is None
                and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "nccl"):
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'. "
                "Run `pip install ray[cgraph]` and check cupy installation.")

    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        from ray.dag import InputNode, MultiOutputNode

        logger.info("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
                    envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE)
        logger.info("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
                    envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM)

        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        if channel_type not in ("auto", "nccl", "shm"):
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'.")

        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        logger.info("RAY_CGRAPH_get_timeout is set to %s",
                    os.environ["RAY_CGRAPH_get_timeout"])  # noqa: SIM112

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # For V0:
            # ExecuteModelRequest -> 0 -> (ExecuteModelReq, IntermediateTensors) -> 4 -> SamplerOutput   # noqa: E501
            # ExecuteModelRequest -> 1 -> (ExecuteModelReq, IntermediateTensors) -> 5 -> SamplerOutput   # noqa: E501
            # ExecuteModelRequest -> 2 -> (ExecuteModelReq, IntermediateTensors) -> 6 -> SamplerOutput   # noqa: E501
            # ExecuteModelRequest -> 3 -> (ExecuteModelReq, IntermediateTensors) -> 7 -> SamplerOutput   # noqa: E501
            #
            # For V1:
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                if self.use_v1:
                    outputs = [
                        worker.execute_model_ray.
                        bind(  # type: ignore[attr-defined]
                            outputs[i]) for i, worker in enumerate(tp_group)
                    ]
                else:
                    outputs = [
                        worker.execute_model_spmd.
                        bind(  # type: ignore[attr-defined]
                            outputs[i]) for i, worker in enumerate(tp_group)
                    ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if (pp_rank < last_pp_rank and
                        envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        for output in outputs
                    ]

            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.
            VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM)

    def __del__(self):
        self.shutdown()

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        # This log will show what virtual_engine LLMEngine.step_async set in the request
        logger.error(
            f"[RAY_EXEC_ENTRY pid={os.getpid()}] Received execute_model_req with "
            f"req.virtual_engine: {execute_model_req.virtual_engine}"
        )

        if not self.use_ray_spmd_worker:
            logger.error(f"[RAY_EXEC_INFO pid={os.getpid()}] use_ray_spmd_worker is False. Calling super().execute_model_async.")
            return await super().execute_model_async(execute_model_req)

        # This path is taken if self.use_ray_spmd_worker is True
        logger.error(f"[RAY_EXEC_INFO pid={os.getpid()}] use_ray_spmd_worker is True. Using Ray DAG.")
        if self.forward_dag is None:
            # _compiled_ray_dag() defines how workers in the DAG receive and process data.
            # This is where the request might be "misinterpreted" or misrouted internally by the DAG
            # if the DAG structure doesn't properly use the virtual_engine from the serialized request
            # for each pipeline stage actor.
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=True)

        # Log again right before encoding to ensure no modifications happened if super() was complex
        # or if there were other preamble steps.
        logger.error(
            f"[RAY_EXEC_PRE_ENCODE pid={os.getpid()}] Before encoding, "
            f"req.virtual_engine: {execute_model_req.virtual_engine}"
        )
        serialized_data = self.input_encoder.encode(execute_model_req)
        
        # The serialized_data (containing the ExecuteModelRequest with its virtual_engine field)
        # is passed to the Ray DAG. Each Ray actor (worker) in the DAG will deserialize this.
        # If a worker for pipeline stage 1 deserializes a request that has virtual_engine=0,
        # it will lead to the error we're seeing.
        dag_future = await self.forward_dag.execute_async(serialized_data)
        output = await dag_future[0]
        return self.output_decoder.decode(output)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        assert not self.use_ray_spmd_worker, (
            "driver_worker does not exist for VLLM_USE_RAY_SPMD_WORKER=1")
        if not self.tp_driver_workers:
            return await self.driver_exec_method("execute_model",
                                                 execute_model_req)
        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(self.driver_exec_method, self.pp_locks[0],
                                    "execute_model", execute_model_req))
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers,
                                                start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(driver_worker.execute_method.remote,
                                        self.pp_locks[pp_rank],
                                        "execute_model", execute_model_req)))

        results = await asyncio.gather(*tasks)

        # Only the last PP stage has the final results.
        return results[-1]

    async def _start_worker_execution_loop(self):
        assert not self.use_ray_spmd_worker, (
            "worker loop is disabled for VLLM_USE_RAY_SPMD_WORKER=1")
        coros = [
            worker.execute_method.remote("start_worker_execution_loop")
            for worker in self.non_driver_workers
        ]
        return await asyncio.gather(*coros)

    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return
