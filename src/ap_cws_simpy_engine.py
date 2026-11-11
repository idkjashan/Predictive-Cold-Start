"""
AP-CWS: Discrete-Event Serverless Workflow Simulation Engine
High-fidelity SimPy-based execution environment for serverless DAG workflows.
Models heterogeneous nodes, container lifecycles (cold starts, warm reuse, keep-alive, eviction),
dynamic memory constraints, network transfer, execution queues, and cost accounting.
"""

import simpy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict


@dataclass
class FunctionSpec:
    """Specification of a serverless function in a workflow DAG."""
    id: str
    memory_mb: int
    cpu_units: float
    base_duration: float  # execution time in seconds on a baseline node (speed=1.0)
    cold_start_delay: float  # container initialization + image pull time (seconds)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Definition of a workflow DAG structure."""
    id: str
    functions: Dict[str, FunctionSpec]
    entry_nodes: List[str]
    exit_nodes: List[str]
    slo_deadline: float  # Target end-to-end duration in seconds


class ContainerInstance:
    """Represents an active or warm container instance on a physical node."""
    def __init__(self, container_id: str, func_id: str, memory_mb: int, node_id: str, creation_time: float):
        self.container_id = container_id
        self.func_id = func_id
        self.memory_mb = memory_mb
        self.node_id = node_id
        self.creation_time = creation_time
        self.last_used_time = creation_time
        self.is_busy = False
        self.invocation_count = 0


class SimulationNode:
    """A compute node in the heterogeneous serverless cluster."""
    def __init__(self, env: simpy.Environment, node_id: str, total_memory_mb: int, speed: float, price_per_hour: float):
        self.env = env
        self.id = node_id
        self.total_memory_mb = total_memory_mb
        self.free_memory_mb = total_memory_mb
        self.speed = speed
        self.price_per_hour = price_per_hour
        # Cost per MB-second based on hourly price
        self.price_per_mb_sec = (price_per_hour / 3600.0) / total_memory_mb

        # Containers hosted on this node: {func_id: [ContainerInstance, ...]}
        self.warm_containers: Dict[str, List[ContainerInstance]] = defaultdict(list)
        # Lock/Resource to serialize node operations if needed
        self.resource = simpy.Resource(env, capacity=1000)

        # Statistics
        self.total_invocations = 0
        self.cold_starts = 0
        self.warm_hits = 0
        self.busy_memory_integral = 0.0  # MB * seconds
        self.idle_memory_integral = 0.0  # MB * seconds
        self.last_update_time = 0.0

    def update_integrals(self):
        """Update time-integral metrics of memory usage."""
        dt = self.env.now - self.last_update_time
        if dt > 0:
            busy_mem = sum(c.memory_mb for c_list in self.warm_containers.values() for c in c_list if c.is_busy)
            idle_mem = sum(c.memory_mb for c_list in self.warm_containers.values() for c in c_list if not c.is_busy)
            self.busy_memory_integral += busy_mem * dt
            self.idle_memory_integral += idle_mem * dt
            self.last_update_time = self.env.now

    def get_idle_container(self, func_id: str) -> Optional[ContainerInstance]:
        """Find an idle warm container for func_id, if any."""
        containers = self.warm_containers.get(func_id, [])
        for c in containers:
            if not c.is_busy:
                return c
        return None

    def allocate_container(self, func_id: str, memory_mb: int) -> Optional[ContainerInstance]:
        """Allocate memory and create a new container instance."""
        self.update_integrals()
        if self.free_memory_mb >= memory_mb:
            self.free_memory_mb -= memory_mb
            c_id = f"{self.id}_{func_id}_{len(self.warm_containers[func_id])}_{self.env.now:.3f}"
            container = ContainerInstance(c_id, func_id, memory_mb, self.id, self.env.now)
            self.warm_containers[func_id].append(container)
            return container
        return None

    def evict_container(self, container: ContainerInstance) -> bool:
        """Evict an idle warm container to reclaim memory."""
        self.update_integrals()
        if not container.is_busy and container in self.warm_containers[container.func_id]:
            self.warm_containers[container.func_id].remove(container)
            self.free_memory_mb += container.memory_mb
            return True
        return False

    def evict_idle_by_policy(self, memory_needed: int, keep_least_recent: bool = True) -> bool:
        """Evict idle containers until memory_needed is freed, or no more can be evicted."""
        self.update_integrals()
        all_idle: List[ContainerInstance] = []
        for c_list in self.warm_containers.values():
            for c in c_list:
                if not c.is_busy:
                    all_idle.append(c)

        if keep_least_recent:
            # Sort by last_used_time ascending (LRU)
            all_idle.sort(key=lambda c: c.last_used_time)

        for c in all_idle:
            if self.free_memory_mb >= memory_needed:
                return True
            self.evict_container(c)

        return self.free_memory_mb >= memory_needed


@dataclass
class WorkflowExecutionRecord:
    """Execution telemetry recorded for a single workflow instance."""
    workflow_instance_id: str
    workflow_id: str
    arrival_time: float
    start_time: float = 0.0
    completion_time: float = 0.0
    end_to_end_latency: float = 0.0
    slo_deadline: float = 0.0
    slo_violated: bool = False
    critical_path_functions: List[str] = field(default_factory=list)
    critical_path_cold_starts: int = 0
    total_cold_starts: int = 0
    total_invocations: int = 0
    function_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_cost_usd: float = 0.0


class ClusterSimulator:
    """
    Coordinates nodes, workflow arrivals, discrete event orchestration,
    and interfaces with scheduling policies.
    """
    def __init__(self, node_configs: List[Dict[str, Any]], random_seed: int = 42):
        self.env = simpy.Environment()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        random.seed(random_seed)

        self.nodes: Dict[str, SimulationNode] = {}
        for cfg in node_configs:
            node = SimulationNode(
                self.env,
                node_id=cfg["id"],
                total_memory_mb=cfg["memory_mb"],
                speed=cfg.get("speed", 1.0),
                price_per_hour=cfg.get("price_per_hour", 0.05)
            )
            self.nodes[node.id] = node

        # Metrics collection
        self.workflow_records: List[WorkflowExecutionRecord] = []
        self.cold_start_events: List[Dict[str, Any]] = []

    def get_cluster_state(self) -> Dict[str, Any]:
        """Snapshot of current cluster resources for scheduling policies."""
        state = {}
        for nid, n in self.nodes.items():
            state[nid] = {
                "free_memory_mb": n.free_memory_mb,
                "total_memory_mb": n.total_memory_mb,
                "speed": n.speed,
                "price_per_hour": n.price_per_hour,
                "warm_counts": {fid: sum(1 for c in clist if not c.is_busy) for fid, clist in n.warm_containers.items()}
            }
        return state

    def simulate_workflow_instance(
        self,
        instance_id: str,
        workflow_def: WorkflowDefinition,
        scheduler_policy: Any,
        critical_path: List[str]
    ):
        """
        SimPy process for executing an entire workflow instance DAG.
        Tracks dependencies, schedules ready tasks, performs prewarming, and logs metrics.
        """
        arrival_time = self.env.now
        record = WorkflowExecutionRecord(
            workflow_instance_id=instance_id,
            workflow_id=workflow_def.id,
            arrival_time=arrival_time,
            slo_deadline=workflow_def.slo_deadline,
            critical_path_functions=critical_path
        )

        completed_functions: Set[str] = set()
        function_events: Dict[str, simpy.Event] = {
            fid: self.env.event() for fid in workflow_def.functions
        }

        # Process for a single function execution within the DAG
        def run_function(func_id: str):
            func_spec = workflow_def.functions[func_id]

            # 1. Wait for all parent functions to complete
            if func_spec.parents:
                parent_events = [function_events[p] for p in func_spec.parents]
                yield simpy.AllOf(self.env, parent_events)

            func_ready_time = self.env.now

            # 2. Query scheduler for placement node
            selected_node_id = scheduler_policy.select_node(
                func_spec=func_spec,
                workflow_def=workflow_def,
                cluster_simulator=self,
                is_critical=(func_id in critical_path)
            )

            target_node = self.nodes.get(selected_node_id, list(self.nodes.values())[0])

            # 3. Trigger just-in-time successor prewarming if scheduler supports it
            if hasattr(scheduler_policy, "notify_function_start"):
                scheduler_policy.notify_function_start(
                    func_spec=func_spec,
                    workflow_def=workflow_def,
                    cluster_simulator=self,
                    target_node_id=target_node.id
                )

            # 4. Check for warm container on selected node
            container = target_node.get_idle_container(func_id)
            is_cold_start = False
            cold_delay = 0.0

            if container is not None:
                # Warm hit!
                container.is_busy = True
                container.last_used_time = self.env.now
                container.invocation_count += 1
                target_node.warm_hits += 1
            else:
                # Cold start!
                is_cold_start = True
                target_node.cold_starts += 1
                record.total_cold_starts += 1
                if func_id in critical_path:
                    record.critical_path_cold_starts += 1

                # If insufficient memory, evict idle containers
                if target_node.free_memory_mb < func_spec.memory_mb:
                    target_node.evict_idle_by_policy(func_spec.memory_mb)

                # If still insufficient memory, wait for memory to be released (queueing)
                wait_start = self.env.now
                while target_node.free_memory_mb < func_spec.memory_mb:
                    yield self.env.timeout(0.05)  # brief wait
                    if self.env.now - wait_start > 10.0:  # Safety timeout
                        break

                container = target_node.allocate_container(func_id, func_spec.memory_mb)
                if container is None:
                    # Fallback allocate forced (oversubscribed briefly)
                    c_id = f"{target_node.id}_{func_id}_forced_{self.env.now:.3f}"
                    container = ContainerInstance(c_id, func_id, func_spec.memory_mb, target_node.id, self.env.now)
                    target_node.warm_containers[func_id].append(container)

                container.is_busy = True
                # Cold start startup delay with minor jitter (±10%)
                jitter = self.rng.uniform(0.9, 1.1)
                cold_delay = func_spec.cold_start_delay * jitter
                yield self.env.timeout(cold_delay)

            target_node.total_invocations += 1
            record.total_invocations += 1

            # 5. Execute function computation
            # Execution duration scaled by node speed with normal variation
            speed_factor = target_node.speed
            base_dur = func_spec.base_duration / speed_factor
            exec_jitter = self.rng.normal(1.0, 0.05)
            exec_duration = max(0.01, base_dur * exec_jitter)

            exec_start_time = self.env.now
            yield self.env.timeout(exec_duration)
            exec_finish_time = self.env.now

            # 6. Release container back to warm pool
            container.is_busy = False
            container.last_used_time = self.env.now
            target_node.update_integrals()

            # Schedule keep-alive expiration check or inform scheduler
            if hasattr(scheduler_policy, "notify_function_completion"):
                scheduler_policy.notify_function_completion(
                    func_spec=func_spec,
                    container=container,
                    cluster_simulator=self
                )

            # Record function metrics
            record.function_records[func_id] = {
                "node_id": target_node.id,
                "is_cold": is_cold_start,
                "cold_delay": cold_delay,
                "exec_duration": exec_duration,
                "total_duration": (exec_finish_time - func_ready_time),
                "is_critical": (func_id in critical_path)
            }

            # AWS Lambda-style cost: $0.0000166667 per GB-second of execution
            gb_seconds = (func_spec.memory_mb / 1024.0) * (exec_duration + cold_delay)
            record.execution_cost_usd += gb_seconds * 0.00001667

            # Mark complete
            completed_functions.add(func_id)
            function_events[func_id].succeed()

        # Launch all functions as concurrent SimPy processes
        for fid in workflow_def.functions:
            self.env.process(run_function(fid))

        # Wait for all exit nodes to complete
        exit_events = [function_events[ex] for ex in workflow_def.exit_nodes]
        yield simpy.AllOf(self.env, exit_events)

        record.completion_time = self.env.now
        record.end_to_end_latency = record.completion_time - record.arrival_time
        record.slo_violated = (record.end_to_end_latency > record.slo_deadline)
        self.workflow_records.append(record)

    def run_workload(
        self,
        workflow_def: WorkflowDefinition,
        scheduler_policy: Any,
        arrival_times: List[float],
        critical_path: List[str]
    ):
        """
        Feeds workflow instances into the simulator according to arrival_times.
        """
        def workload_generator():
            for idx, arr_time in enumerate(arrival_times):
                delay = arr_time - self.env.now
                if delay > 0:
                    yield self.env.timeout(delay)

                inst_id = f"wf_{idx}_{self.env.now:.2f}"
                self.env.process(
                    self.simulate_workflow_instance(
                        instance_id=inst_id,
                        workflow_def=workflow_def,
                        scheduler_policy=scheduler_policy,
                        critical_path=critical_path
                    )
                )

        self.env.process(workload_generator())

        # Also launch periodic pool manager process if scheduler supports it
        if hasattr(scheduler_policy, "periodic_pool_management"):
            def pool_manager_loop():
                while True:
                    yield self.env.timeout(scheduler_policy.planning_interval)
                    scheduler_policy.periodic_pool_management(self, workflow_def)
            self.env.process(pool_manager_loop())

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Aggregate all metrics into final evaluation dictionary."""
        for n in self.nodes.values():
            n.update_integrals()

        total_workflows = len(self.workflow_records)
        if total_workflows == 0:
            return {}

        latencies = [r.end_to_end_latency for r in self.workflow_records]
        total_invocs = sum(r.total_invocations for r in self.workflow_records)
        total_colds = sum(r.total_cold_starts for r in self.workflow_records)

        crit_invocs = sum(
            1 for r in self.workflow_records for fid, f in r.function_records.items() if f["is_critical"]
        )
        crit_colds = sum(r.critical_path_cold_starts for r in self.workflow_records)

        slo_violations = sum(1 for r in self.workflow_records if r.slo_violated)
        total_cost = sum(r.execution_cost_usd for r in self.workflow_records)

        # Total idle memory GB-hours
        total_idle_mb_sec = sum(n.idle_memory_integral for n in self.nodes.values())
        idle_gb_hours = (total_idle_mb_sec / 1024.0) / 3600.0

        # Provider idle holding cost
        idle_holding_cost = sum(
            (n.idle_memory_integral * n.price_per_mb_sec) for n in self.nodes.values()
        )

        return {
            "total_workflows": total_workflows,
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
            "cold_start_rate_pct": (total_colds / max(1, total_invocs)) * 100.0,
            "critical_path_cold_start_pct": (crit_colds / max(1, crit_invocs)) * 100.0,
            "slo_violation_rate_pct": (slo_violations / total_workflows) * 100.0,
            "execution_cost_usd": total_cost,
            "idle_gb_hours": idle_gb_hours,
            "total_cost_usd": total_cost + idle_holding_cost,
        }
