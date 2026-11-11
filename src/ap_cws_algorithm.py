"""
AP-CWS: Predictive, Cold-Start-Aware Workflow Scheduling for Serverless Computing
Complete unified algorithm implementation
"""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from datetime import datetime, timedelta


@dataclass
class Function:
    """Serverless function with attributes"""
    id: str
    memory: int  # MB
    cpu: float  # normalized units
    avg_duration: float  # seconds
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class Node:
    """Compute node in the cluster"""
    id: str
    total_memory: int  # MB
    speed: float  # relative to baseline
    price_per_hour: float
    free_memory: int = field(init=False)

    def __post_init__(self):
        self.free_memory = self.total_memory


@dataclass
class WorkflowDAG:
    """Directed Acyclic Graph of functions"""
    id: str
    functions: Dict[str, Function]
    entry_nodes: List[str]
    exit_nodes: List[str]
    est: Dict[str, float] = field(default_factory=dict)  # Earliest Start Time
    lst: Dict[str, float] = field(default_factory=dict)  # Latest Start Time
    slack: Dict[str, float] = field(default_factory=dict)


class HoltForecast:
    """Holt's linear exponential smoothing for demand forecasting"""

    def __init__(self, alpha: float = 0.2, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = 0
        self.trend = 0
        self.history = []

    def update(self, demand: int):
        """Update forecast with new demand observation"""
        if not self.history:
            self.level = demand
            self.trend = 0
        else:
            prev_level = self.level
            self.level = self.alpha * demand + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

        self.history.append(demand)
        return self.level, self.trend

    def forecast(self, steps: int = 1) -> float:
        """Forecast demand `steps` ahead"""
        return self.level + steps * self.trend


class ColdStartModel:
    """Model cold-start probability based on warm pool size"""

    def __init__(self, baseline_cold_starts: int, max_cold_starts: int):
        self.baseline = baseline_cold_starts
        self.max = max_cold_starts

    def probability(self, pool_size: int) -> float:
        """
        Estimate P(cold start | pool size W)
        Decreasing sigmoid: as W increases, cold-start risk decreases
        """
        if pool_size == 0:
            return 1.0
        # Exponential decay model
        decay = np.exp(-0.3 * pool_size)
        return min(self.baseline * decay, 1.0)

    def pool_size_for_target(self, target_prob: float) -> int:
        """Find minimum pool size W to achieve target cold-start probability"""
        if target_prob >= 1.0:
            return 0
        w = 0
        while self.probability(w) > target_prob and w < 100:
            w += 1
        return w


class APCWSScheduler:
    """AP-CWS: Unified pool sizing and function placement"""

    def __init__(
        self,
        nodes: List[Node],
        w_min: int = 1,
        w_max: int = 20,
        planning_interval: float = 60.0,  # seconds
        slo_threshold: float = 0.95,  # 95th percentile latency target
    ):
        self.nodes = nodes
        self.w_min = w_min
        self.w_max = w_max
        self.planning_interval = planning_interval
        self.slo_threshold = slo_threshold

        # State
        self.warm_pools: Dict[str, Dict[str, int]] = {}  # {func_id: {node_id: count}}
        self.forecasts: Dict[str, HoltForecast] = {}
        self.cold_start_models: Dict[str, ColdStartModel] = {}
        self.placement: Dict[str, str] = {}  # {func_id: node_id}
        self.metrics = defaultdict(list)

    def calculate_criticality(self, dag: WorkflowDAG, time_budget: float) -> Dict[str, float]:
        """
        Calculate criticality K for each function.
        K = 1 - (LST - EST) / T*
        K = 1: on critical path (no slack)
        K = 0: maximum slack
        """
        criticality = {}
        for func_id in dag.functions:
            est = dag.est.get(func_id, 0)
            lst = dag.lst.get(func_id, time_budget)
            slack = lst - est

            if slack <= 0 or time_budget == 0:
                k = 1.0  # critical path
            else:
                k = 1.0 - (slack / time_budget)
            criticality[func_id] = max(0, min(1, k))
        return criticality

    def compute_est_lst(self, dag: WorkflowDAG, time_budget: Optional[float] = None):
        """Compute EST and LST via topological passes"""
        # Forward pass: EST
        in_degree = defaultdict(int)
        for func_id in dag.functions:
            for child in dag.functions[func_id].children:
                in_degree[child] += 1

        queue = deque(dag.entry_nodes)
        est = {node: 0 for node in dag.entry_nodes}

        while queue:
            func_id = queue.popleft()
            func = dag.functions[func_id]

            for child in func.children:
                child_func = dag.functions[child]
                # EST(child) = max(EST(parent) + duration(parent))
                child_est = est.get(child, 0)
                est[child] = max(child_est, est[func_id] + func.avg_duration)

                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        dag.est = est
        total_time = max(est[node] for node in dag.exit_nodes)

        # Backward pass: LST
        lst = {node: total_time for node in dag.exit_nodes}
        queue = deque(dag.exit_nodes)

        while queue:
            func_id = queue.popleft()
            func = dag.functions[func_id]

            for parent_id in func.parents:
                parent = dag.functions[parent_id]
                # LST(parent) = min(LST(child) - duration(parent))
                parent_lst = lst.get(parent_id, total_time)
                lst[parent_id] = min(parent_lst, lst[func_id] - parent.avg_duration)

                if parent_id not in queue:
                    queue.append(parent_id)

        dag.lst = lst
        return total_time

    def size_pools(
        self,
        dag: WorkflowDAG,
        invocation_history: Dict[str, List[int]],
        target_cold_prob: float = 0.05,
    ) -> Dict[str, int]:
        """
        Algorithm 1, lines 3–11: Size warm pools per function
        """
        # 1. Forecast demand
        pool_sizes = {}
        for func_id in dag.functions:
            if func_id not in self.forecasts:
                self.forecasts[func_id] = HoltForecast()
                self.cold_start_models[func_id] = ColdStartModel(
                    baseline_cold_starts=5, max_cold_starts=20
                )

            history = invocation_history.get(func_id, [1, 1, 1])
            level, trend = self.forecasts[func_id].update(history[-1] if history else 1)

            # 2. Determine cold-start probability and pool size
            model = self.cold_start_models[func_id]
            w = model.pool_size_for_target(target_cold_prob)
            pool_sizes[func_id] = max(self.w_min, min(self.w_max, w))

        return pool_sizes

    def place_functions(
        self,
        dag: WorkflowDAG,
        criticality: Dict[str, float],
        pool_sizes: Dict[str, int],
        kappa: float = 0.5,  # threshold for just-in-time prewarming
    ) -> Dict[str, str]:
        """
        Algorithm 1, lines 12–17: Dispatch functions to nodes
        Joint score S combines criticality, warm placement, and cost
        """
        placement = {}
        ready_queue = deque(dag.entry_nodes)
        processed = set()

        # Ready-queue sorted by criticality (descending)
        ready_queue = sorted(dag.entry_nodes, key=lambda f: criticality[f], reverse=True)

        for func_id in ready_queue:
            func = dag.functions[func_id]
            k = criticality[func_id]

            # Filter nodes with enough free memory
            capable_nodes = [
                node for node in self.nodes if node.free_memory >= func.memory
            ]

            if not capable_nodes:
                continue

            # Score each node
            best_node = None
            best_score = float("inf")

            for node in capable_nodes:
                # y = 0 if warm, 1 if cold
                has_warm = func_id in self.warm_pools and self.warm_pools[func_id].get(
                    node.id, 0
                ) > 0
                y = 0 if has_warm else 1

                # Scoring components
                startup_penalty = y * func.avg_duration * 0.5  # cold start latency
                memory_cost = func.memory / 1024  # normalized
                utilization_benefit = -0.1 * (node.free_memory / node.total_memory)

                # Overall score: criticality and node state
                score = (1 + k) * (y * startup_penalty + 0.01 * memory_cost) + utilization_benefit

                if score < best_score:
                    best_score = score
                    best_node = node

            if best_node:
                placement[func_id] = best_node.id
                best_node.free_memory -= func.memory
                processed.add(func_id)

        return placement

    def prewarm_successors(
        self,
        dag: WorkflowDAG,
        placement: Dict[str, str],
        criticality: Dict[str, float],
        kappa: float = 0.5,
    ):
        """
        Algorithm 1, lines 18–19: Prewarming critical successors
        """
        for func_id, node_id in placement.items():
            for child_id in dag.functions[func_id].children:
                child_k = criticality[child_id]

                # Just-in-time prewarm if child is critical
                if child_k >= kappa:
                    node = next(n for n in self.nodes if n.id == node_id)
                    if child_id not in self.warm_pools:
                        self.warm_pools[child_id] = defaultdict(int)
                    if node.free_memory >= dag.functions[child_id].memory:
                        self.warm_pools[child_id][node_id] += 1
                        node.free_memory -= dag.functions[child_id].memory

    def plan(
        self,
        dag: WorkflowDAG,
        invocation_history: Dict[str, List[int]],
        slo_latency: float,
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Execute planning phase: size pools and place functions
        """
        # Compute EST/LST for criticality
        total_time = self.compute_est_lst(dag, slo_latency)

        # Calculate criticality K
        criticality = self.calculate_criticality(dag, total_time)

        # Size pools (lines 3–11)
        pool_sizes = self.size_pools(dag, invocation_history, target_cold_prob=0.05)

        # Place functions and prewarm (lines 12–19)
        placement = self.place_functions(dag, criticality, pool_sizes, kappa=0.5)
        self.prewarm_successors(dag, placement, criticality, kappa=0.5)

        return pool_sizes, placement

    def execute(
        self,
        dag: WorkflowDAG,
        placement: Dict[str, str],
        simulated_exec_time: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Simulate execution: record cold starts, latency, cost
        """
        results = {
            "cold_starts": 0,
            "total_latency": 0,
            "idle_memory": 0,
            "cost": 0,
            "slo_violations": 0,
        }

        for func_id, node_id in placement.items():
            node = next(n for n in self.nodes if n.id == node_id)

            # Check if warm
            has_warm = (
                func_id in self.warm_pools and self.warm_pools[func_id].get(node_id, 0) > 0
            )

            if not has_warm:
                results["cold_starts"] += 1
                # Cold-start penalty
                actual_time = simulated_exec_time[func_id] + 1.0
            else:
                actual_time = simulated_exec_time[func_id]

            results["total_latency"] += actual_time

        # Calculate costs and idle memory
        for node in self.nodes:
            idle = node.free_memory
            results["idle_memory"] += idle
            results["cost"] += node.price_per_hour * (1 - idle / node.total_memory)

        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_workflow():
    """Create a simple DAG for demonstration"""
    functions = {
        "f1": Function("f1", memory=256, cpu=0.5, avg_duration=0.5, parents=[], children=["f2", "f3"]),
        "f2": Function("f2", memory=512, cpu=1.0, avg_duration=1.0, parents=["f1"], children=["f4"]),
        "f3": Function("f3", memory=256, cpu=0.5, avg_duration=0.3, parents=["f1"], children=["f4"]),
        "f4": Function("f4", memory=1024, cpu=2.0, avg_duration=2.0, parents=["f2", "f3"], children=[]),
    }
    dag = WorkflowDAG(
        id="workflow_1",
        functions=functions,
        entry_nodes=["f1"],
        exit_nodes=["f4"],
    )
    return dag


def main():
    """Run AP-CWS scheduler"""
    # Setup nodes
    nodes = [
        Node("node_1", total_memory=4096, speed=1.0, price_per_hour=0.05),
        Node("node_2", total_memory=4096, speed=1.2, price_per_hour=0.06),
        Node("node_3", total_memory=8192, speed=2.0, price_per_hour=0.10),
    ]

    # Create scheduler
    scheduler = APCWSScheduler(nodes, w_min=1, w_max=10)

    # Setup workflow
    dag = example_workflow()

    # Mock invocation history
    invocation_history = {
        "f1": [100, 110, 105, 115],
        "f2": [50, 55, 52, 60],
        "f3": [45, 50, 48, 55],
        "f4": [40, 42, 41, 45],
    }

    # Planning phase
    pool_sizes, placement = scheduler.plan(dag, invocation_history, slo_latency=5.0)

    print("=== AP-CWS Unified Scheduling ===")
    print(f"\nPool Sizes (W): {pool_sizes}")
    print(f"Function Placement: {placement}")

    # Simulate execution
    simulated_exec_times = {f: dag.functions[f].avg_duration for f in dag.functions}
    results = scheduler.execute(dag, placement, simulated_exec_times)

    print(f"\n=== Execution Results ===")
    for key, val in results.items():
        print(f"{key}: {val:.2f}")


if __name__ == "__main__":
    main()
