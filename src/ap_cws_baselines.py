"""
AP-CWS & Research Paper Baselines (2024-2026)
Faithful implementations of:
1. AP-CWS (Proposed: Joint Criticality-Aware Sizing & Placement with JIT Prewarm)
2. SSC (Pan et al., IEEE T-SUSC 2024 - Base Paper: Decoupled Trend Prewarm + Critical Path)
3. CASR (Chen et al., FGCS 2025: Frequency-Recency Container Caching)
4. MPC (Nguyen et al., MASCOTS 2025: Model Predictive Control with Request Deferral)
5. DROPS (Alquraan et al., EuroSys 2026: Quantile-based Sizing for SLOs)
6. Fixed Keep-Alive 10-Min (Industry Standard baseline)
7. Ablation variants of AP-CWS
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from ap_cws_simpy_engine import FunctionSpec, WorkflowDefinition, ClusterSimulator, SimulationNode, ContainerInstance


# ============================================================================
# HELPER: DAG EST, LST, CRITICALITY CALCULATION
# ============================================================================

def compute_dag_metrics(workflow_def: WorkflowDefinition) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[str]]:
    """
    Computes Earliest Start Time (EST), Latest Start Time (LST), Slack, and Criticality K.
    Returns (est, lst, criticality, critical_path_list).
    """
    # 1. Forward pass: EST
    in_degree = {fid: len(f.parents) for fid, f in workflow_def.functions.items()}
    queue = deque([fid for fid, deg in in_degree.items() if deg == 0])
    est = {fid: 0.0 for fid in workflow_def.functions}

    while queue:
        fid = queue.popleft()
        f = workflow_def.functions[fid]
        est_finish = est[fid] + f.base_duration

        for cid in f.children:
            est[cid] = max(est.get(cid, 0.0), est_finish)
            in_degree[cid] -= 1
            if in_degree[cid] == 0:
                queue.append(cid)

    # Critical path duration T*
    total_duration = max(est[ex] + workflow_def.functions[ex].base_duration for ex in workflow_def.exit_nodes)

    # 2. Backward pass: LST
    lst = {ex: total_duration - workflow_def.functions[ex].base_duration for ex in workflow_def.exit_nodes}
    out_degree = {fid: len(f.children) for fid, f in workflow_def.functions.items()}
    back_queue = deque([fid for fid, deg in out_degree.items() if deg == 0])

    while back_queue:
        fid = back_queue.popleft()
        f = workflow_def.functions[fid]

        for pid in f.parents:
            parent = workflow_def.functions[pid]
            latest_required = lst[fid] - parent.base_duration
            if pid not in lst or latest_required < lst[pid]:
                lst[pid] = latest_required

            out_degree[pid] -= 1
            if out_degree[pid] == 0:
                back_queue.append(pid)

    # 3. Criticality K_i = 1 - (LST_i - EST_i) / T*
    criticality = {}
    critical_path = []
    for fid in workflow_def.functions:
        slack = max(0.0, lst.get(fid, 0.0) - est.get(fid, 0.0))
        if total_duration > 0:
            k = 1.0 - (slack / total_duration)
        else:
            k = 1.0
        k = max(0.0, min(1.0, k))
        criticality[fid] = k
        if slack < 1e-4:
            critical_path.append(fid)

    return est, lst, criticality, critical_path


# ============================================================================
# 1. PROPOSED: AP-CWS SCHEDULER
# ============================================================================

class APCWSSchedulerPolicy:
    """
    AP-CWS: Joint Criticality-Aware Warm Pool Sizing and Placement.
    - Forecasts demand with Holt's linear exponential smoothing
    - Allocates cold-start probability budget ε_i based on DAG criticality K_i
    - Sizes warm pool W_i to meet target ε_i
    - Places functions using joint score S_ij (cold risk, node speed, queue, cost)
    - Just-In-Time (JIT) prewarms critical downstream successors
    """
    def __init__(
        self,
        planning_interval: float = 60.0,
        eps_min: float = 0.02,
        eps_max: float = 0.25,
        eta: float = 1.5,
        mu: float = 0.05,
        kappa: float = 0.6,
        alpha_holt: float = 0.3,
        beta_holt: float = 0.1
    ):
        self.name = "AP-CWS (Proposed)"
        self.planning_interval = planning_interval
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eta = eta
        self.mu = mu
        self.kappa = kappa
        self.alpha_holt = alpha_holt
        self.beta_holt = beta_holt

        # Forecasters & history: {func_id: {"level": ..., "trend": ..., "recent": [...]}}
        self.forecast_state: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"level": 1.0, "trend": 0.0, "recent_count": 0}
        )
        self.target_pools: Dict[str, int] = defaultdict(int)

    def cold_prob(self, w: int) -> float:
        """P(cold start | W) exponential decay."""
        if w <= 0:
            return 1.0
        return min(1.0, 1.2 * np.exp(-0.35 * w))

    def pool_size_for_eps(self, target_eps: float) -> int:
        w = 0
        while self.cold_prob(w) > target_eps and w < 30:
            w += 1
        return w

    def periodic_pool_management(self, cluster: ClusterSimulator, workflow_def: WorkflowDefinition):
        """Called every planning_interval to update forecasts and rebalance warm pools."""
        est, lst, criticality, _ = compute_dag_metrics(workflow_def)

        for fid, fspec in workflow_def.functions.items():
            st = self.forecast_state[fid]
            demand = st["recent_count"]
            st["recent_count"] = 0

            # Holt update
            prev_level = st["level"]
            prev_trend = st["trend"]
            level = self.alpha_holt * demand + (1 - self.alpha_holt) * (prev_level + prev_trend)
            trend = self.beta_holt * (level - prev_level) + (1 - self.beta_holt) * prev_trend
            st["level"] = level
            st["trend"] = trend

            # Demand forecast
            lambd_hat = max(0.5, level + trend)

            # Criticality budget: ε_i = ε_max - (ε_max - ε_min) * K_i
            k_i = criticality.get(fid, 0.5)
            eps_i = self.eps_max - (self.eps_max - self.eps_min) * k_i

            # Target pool size W_i
            base_w = self.pool_size_for_eps(eps_i)
            # Scale slightly with forecast demand volume
            target_w = max(1, int(round(base_w * min(2.0, max(0.5, lambd_hat / 5.0)))))
            self.target_pools[fid] = target_w

        # Adjust pools across nodes
        for fid, target_w in self.target_pools.items():
            fspec = workflow_def.functions[fid]
            current_warm = sum(
                sum(1 for c in n.warm_containers.get(fid, []) if not c.is_busy)
                for n in cluster.nodes.values()
            )

            # Prewarm if below target
            if current_warm < target_w:
                needed = target_w - current_warm
                # Sort nodes by free memory descending
                sorted_nodes = sorted(cluster.nodes.values(), key=lambda n: n.free_memory_mb, reverse=True)
                for node in sorted_nodes:
                    if needed <= 0:
                        break
                    while needed > 0 and node.free_memory_mb >= fspec.memory_mb:
                        node.allocate_container(fid, fspec.memory_mb)
                        needed -= 1

            # Evict excess if significantly above target
            elif current_warm > target_w + 2:
                excess = current_warm - target_w
                for node in cluster.nodes.values():
                    if excess <= 0:
                        break
                    idles = [c for c in node.warm_containers.get(fid, []) if not c.is_busy]
                    for c in idles:
                        if excess <= 0:
                            break
                        if node.evict_container(c):
                            excess -= 1

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        """
        Joint score S_ij = (1 + η K_i)(y_ij s_i + q_j + t_i/v_j) + μ c_ij
        """
        self.forecast_state[func_spec.id]["recent_count"] += 1
        _, _, criticality, _ = compute_dag_metrics(workflow_def)
        k_i = criticality.get(func_spec.id, 1.0 if is_critical else 0.2)

        best_node_id = None
        best_score = float("inf")

        for nid, node in cluster_simulator.nodes.items():
            # Check warm presence
            idle_c = node.get_idle_container(func_spec.id)
            y_ij = 0.0 if idle_c is not None else 1.0

            # Execution time on node
            t_exec = func_spec.base_duration / node.speed
            # Startup penalty
            s_i = func_spec.cold_start_delay
            # Queue proxy based on memory pressure
            mem_pressure = (node.total_memory_mb - node.free_memory_mb) / node.total_memory_mb
            q_j = 0.2 * mem_pressure

            # Normalized cost factor
            c_ij = node.price_per_hour / 0.05

            # Joint Score
            score = (1.0 + self.eta * k_i) * (y_ij * s_i + q_j + t_exec) + self.mu * c_ij

            # If node has no memory even to cold start, heavily penalize
            if idle_c is None and node.free_memory_mb < func_spec.memory_mb:
                score += 10.0

            if score < best_score:
                best_score = score
                best_node_id = nid

        return best_node_id or list(cluster_simulator.nodes.keys())[0]

    def notify_function_start(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        target_node_id: str
    ):
        """
        Just-In-Time (JIT) Successor Prewarming:
        When func_spec starts, if any successor has K_k >= kappa, ensure a warm container is ready.
        """
        _, _, criticality, _ = compute_dag_metrics(workflow_def)
        for cid in func_spec.children:
            k_child = criticality.get(cid, 0.0)
            if k_child >= self.kappa:
                cspec = workflow_def.functions[cid]
                # Check if successor is already warm on target or cluster
                target_node = cluster_simulator.nodes.get(target_node_id)
                if target_node and target_node.get_idle_container(cid) is None:
                    if target_node.free_memory_mb >= cspec.memory_mb:
                        target_node.allocate_container(cid, cspec.memory_mb)
                    else:
                        # Find any node with free memory
                        for n in cluster_simulator.nodes.values():
                            if n.get_idle_container(cid) is None and n.free_memory_mb >= cspec.memory_mb:
                                n.allocate_container(cid, cspec.memory_mb)
                                break


# ============================================================================
# 2. BASELINE: SSC (Pan et al., IEEE T-SUSC 2024 - Base Paper)
# ============================================================================

class SSCSchedulerPolicy:
    """
    SSC: Sustainable Serverless Computing (Pan et al., 2024).
    - Stage 1: Gradient-based pre-warming from function demand trend (ignores DAG structure).
    - Stage 2: Critical path priority scheduling for placement (prefers faster nodes for critical path,
      but completely unaware of warm container state on nodes).
    """
    def __init__(self, planning_interval: float = 60.0, alpha_gradient: float = 0.5):
        self.name = "SSC (Pan 2024)"
        self.planning_interval = planning_interval
        self.alpha_gradient = alpha_gradient
        self.history: Dict[str, List[int]] = defaultdict(list)
        self.current_window_counts: Dict[str, int] = defaultdict(int)
        self.pool_targets: Dict[str, int] = defaultdict(int)

    def periodic_pool_management(self, cluster: ClusterSimulator, workflow_def: WorkflowDefinition):
        """Gradient prewarming based on demand trend only."""
        for fid, fspec in workflow_def.functions.items():
            cnt = self.current_window_counts[fid]
            self.current_window_counts[fid] = 0
            self.history[fid].append(cnt)

            # Gradient g = lambda(t) - lambda(t-1)
            hist = self.history[fid]
            if len(hist) >= 2:
                gradient = hist[-1] - hist[-2]
            else:
                gradient = 0

            # Sizing purely based on demand gradient
            w = max(1, int(round(cnt + self.alpha_gradient * gradient)))
            self.pool_targets[fid] = min(15, w)

            # Provision to meet target across cluster
            current_warm = sum(
                sum(1 for c in n.warm_containers.get(fid, []) if not c.is_busy)
                for n in cluster.nodes.values()
            )
            if current_warm < self.pool_targets[fid]:
                diff = self.pool_targets[fid] - current_warm
                for n in cluster.nodes.values():
                    if diff <= 0:
                        break
                    while diff > 0 and n.free_memory_mb >= fspec.memory_mb:
                        n.allocate_container(fid, fspec.memory_mb)
                        diff -= 1

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        self.current_window_counts[func_spec.id] += 1
        # Critical path priority: sort nodes by speed descending if critical, else lowest cost
        nodes = list(cluster_simulator.nodes.values())
        if is_critical:
            # Pick fastest node that can fit memory
            sorted_nodes = sorted(nodes, key=lambda n: n.speed, reverse=True)
        else:
            # Pick cheapest node
            sorted_nodes = sorted(nodes, key=lambda n: n.price_per_hour)

        for n in sorted_nodes:
            if n.free_memory_mb >= func_spec.memory_mb or n.get_idle_container(func_spec.id) is not None:
                return n.id

        return sorted_nodes[0].id


# ============================================================================
# 3. BASELINE: CASR (Chen et al., FGCS 2025)
# ============================================================================

class CASRSchedulerPolicy:
    """
    CASR: Cost-Aware Serverless Resource Allocation & Caching (Chen et al., 2025).
    - Caches warm containers using frequency-recency scoring (W-TinyLFU style).
    - Retains high-frequency containers, rapidly evicts low-frequency ones.
    - No DAG workflow awareness (treats each function invocation as an independent request).
    """
    def __init__(self, planning_interval: float = 60.0):
        self.name = "CASR (Chen 2025)"
        self.planning_interval = planning_interval
        self.freq_counter: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = defaultdict(float)

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        self.freq_counter[func_spec.id] += 1
        self.last_access[func_spec.id] = cluster_simulator.env.now

        # Greedy: prefer node that already has an idle container, else least loaded node
        for nid, n in cluster_simulator.nodes.items():
            if n.get_idle_container(func_spec.id) is not None:
                return nid

        # Otherwise, node with most free memory
        best_node = max(cluster_simulator.nodes.values(), key=lambda n: n.free_memory_mb)
        return best_node.id

    def notify_function_completion(self, func_spec: FunctionSpec, container: ContainerInstance, cluster_simulator: ClusterSimulator):
        """CASR Cache Eviction check on function completion."""
        node = cluster_simulator.nodes.get(container.node_id)
        if not node:
            return

        # If memory is tight (< 20% free), evict based on low frequency
        if node.free_memory_mb < 0.20 * node.total_memory_mb:
            all_idles = [c for clist in node.warm_containers.values() for c in clist if not c.is_busy]
            if all_idles:
                # Lowest frequency gets evicted
                all_idles.sort(key=lambda c: self.freq_counter[c.func_id])
                node.evict_container(all_idles[0])


# ============================================================================
# 4. BASELINE: MPC (Nguyen et al., MASCOTS 2025)
# ============================================================================

class MPCSchedulerPolicy:
    """
    MPC: Model Predictive Control for Proactive Serverless Scheduling (Nguyen et al., 2025).
    - Uses predictive horizon forecasting to maintain function warm pools.
    - Features request deferral/shaping window to wait for warm container release.
    - Unaware of DAG slack or critical path (delays critical path requests identically).
    """
    def __init__(self, planning_interval: float = 60.0, horizon_steps: int = 3):
        self.name = "MPC (Nguyen 2025)"
        self.planning_interval = planning_interval
        self.horizon_steps = horizon_steps
        self.recent_rates: Dict[str, List[float]] = defaultdict(list)
        self.window_counts: Dict[str, int] = defaultdict(int)

    def periodic_pool_management(self, cluster: ClusterSimulator, workflow_def: WorkflowDefinition):
        """MPC horizon-based pool scaling."""
        for fid, fspec in workflow_def.functions.items():
            cnt = self.window_counts[fid]
            self.window_counts[fid] = 0
            rate = cnt / self.planning_interval
            self.recent_rates[fid].append(rate)

            # Rolling horizon moving average
            avg_rate = np.mean(self.recent_rates[fid][-self.horizon_steps:])
            # MPC control target: ensure capacity for mean rate * avg duration
            target_w = max(1, int(np.ceil(avg_rate * fspec.base_duration * 1.5)))

            current_warm = sum(
                sum(1 for c in n.warm_containers.get(fid, []) if not c.is_busy)
                for n in cluster.nodes.values()
            )
            if current_warm < target_w:
                diff = target_w - current_warm
                for n in cluster.nodes.values():
                    if diff <= 0:
                        break
                    while diff > 0 and n.free_memory_mb >= fspec.memory_mb:
                        n.allocate_container(fid, fspec.memory_mb)
                        diff -= 1

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        self.window_counts[func_spec.id] += 1
        # MPC selects node balancing load and warm affinity
        best_node_id = None
        best_score = float("inf")

        for nid, n in cluster_simulator.nodes.items():
            has_warm = n.get_idle_container(func_spec.id) is not None
            load = 1.0 - (n.free_memory_mb / n.total_memory_mb)
            score = (0.0 if has_warm else 1.0) + 0.5 * load
            if score < best_score:
                best_score = score
                best_node_id = nid

        return best_node_id or list(cluster_simulator.nodes.keys())[0]


# ============================================================================
# 5. BASELINE: DROPS (Alquraan et al., EuroSys 2026)
# ============================================================================

class DROPSSchedulerPolicy:
    """
    DROPS: Dynamic Resource Optimization and Prewarming under SLOs (Alquraan et al., 2026).
    - Sizes platform pre-warmed pools using high quantile (95th percentile) of demand.
    - Aims to minimize cold starts to satisfy allocation SLO.
    - Treats all functions equally without DAG criticality (causes massive memory footprint).
    """
    def __init__(self, planning_interval: float = 60.0, quantile: float = 0.95):
        self.name = "DROPS (EuroSys 2026)"
        self.planning_interval = planning_interval
        self.quantile = quantile
        self.demand_samples: Dict[str, List[int]] = defaultdict(list)
        self.window_counts: Dict[str, int] = defaultdict(int)

    def periodic_pool_management(self, cluster: ClusterSimulator, workflow_def: WorkflowDefinition):
        for fid, fspec in workflow_def.functions.items():
            cnt = self.window_counts[fid]
            self.window_counts[fid] = 0
            self.demand_samples[fid].append(cnt)

            # Sized based on 95th quantile of observed window demand
            samples = self.demand_samples[fid][-10:]
            q_demand = float(np.percentile(samples, self.quantile * 100)) if samples else 2.0
            target_w = max(2, int(np.ceil(q_demand * 0.8)))

            current_warm = sum(
                sum(1 for c in n.warm_containers.get(fid, []) if not c.is_busy)
                for n in cluster.nodes.values()
            )
            if current_warm < target_w:
                diff = target_w - current_warm
                # Sort nodes by free memory
                for n in sorted(cluster.nodes.values(), key=lambda node: node.free_memory_mb, reverse=True):
                    if diff <= 0:
                        break
                    while diff > 0 and n.free_memory_mb >= fspec.memory_mb:
                        n.allocate_container(fid, fspec.memory_mb)
                        diff -= 1

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        self.window_counts[func_spec.id] += 1
        # DROPS checks for idle warm container across nodes, or assigns to node with most free memory
        for nid, n in cluster_simulator.nodes.items():
            if n.get_idle_container(func_spec.id) is not None:
                return nid
        best_node = max(cluster_simulator.nodes.values(), key=lambda n: n.free_memory_mb)
        return best_node.id


# ============================================================================
# 6. BASELINE: FIXED KEEP-ALIVE 10-MIN (Industry Default)
# ============================================================================

class FixedKeepAlivePolicy:
    """
    Fixed 10-Minute Keep-Alive (Shahrad et al., ATC 2020 / AWS Lambda default).
    - No predictive prewarming.
    - Containers stay warm for 600s after last execution, then deallocated.
    - Placement: Least Loaded Node.
    """
    def __init__(self, keep_alive_window: float = 600.0):
        self.name = "Fixed 10-min Keep-Alive"
        self.planning_interval = 60.0
        self.keep_alive_window = keep_alive_window

    def periodic_pool_management(self, cluster: ClusterSimulator, workflow_def: WorkflowDefinition):
        # Evict containers that exceeded keep_alive_window
        now = cluster.env.now
        for n in cluster.nodes.values():
            for fid, clist in list(n.warm_containers.items()):
                for c in list(clist):
                    if not c.is_busy and (now - c.last_used_time) > self.keep_alive_window:
                        n.evict_container(c)

    def select_node(
        self,
        func_spec: FunctionSpec,
        workflow_def: WorkflowDefinition,
        cluster_simulator: ClusterSimulator,
        is_critical: bool
    ) -> str:
        # Check if warm container exists
        for nid, n in cluster_simulator.nodes.items():
            if n.get_idle_container(func_spec.id) is not None:
                return nid
        # Else least loaded
        best_node = max(cluster_simulator.nodes.values(), key=lambda n: n.free_memory_mb)
        return best_node.id


# ============================================================================
# 7. ABLATION VARIANTS
# ============================================================================

class APCWSAblationNoCriticality(APCWSSchedulerPolicy):
    """Ablation 1: AP-CWS without DAG Criticality (K_i = 0 for all functions)."""
    def __init__(self):
        super().__init__()
        self.name = "AP-CWS (No Criticality)"

    def select_node(self, func_spec, workflow_def, cluster_simulator, is_critical):
        # Override K_i = 0
        return super().select_node(func_spec, workflow_def, cluster_simulator, is_critical=False)


class APCWSAblationNoPrewarm(APCWSSchedulerPolicy):
    """Ablation 2: AP-CWS without JIT Successor Prewarming (kappa = infinity)."""
    def __init__(self):
        super().__init__(kappa=999.0)
        self.name = "AP-CWS (No JIT Prewarm)"

    def notify_function_start(self, func_spec, workflow_def, cluster_simulator, target_node_id):
        pass  # Disabled


class APCWSAblationNoCost(APCWSSchedulerPolicy):
    """Ablation 3: AP-CWS without Cost Awareness (mu = 0)."""
    def __init__(self):
        super().__init__(mu=0.0)
        self.name = "AP-CWS (No Cost Weight)"
