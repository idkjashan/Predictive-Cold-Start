"""
AP-CWS Enhanced: Integration with optimization tools and monitoring
Production-ready with Prometheus metrics, hyperparameter tuning, and ML forecasting
"""

import os
import sys
import json
import time

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import threading
from abc import ABC, abstractmethod


# ============================================================================
# PROMETHEUS METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """Prometheus-compatible metrics collector"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
        self.timestamp = datetime.now()

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric with optional labels"""
        with self.lock:
            self.metrics[name].append({
                "value": value,
                "timestamp": datetime.now(),
                "labels": labels or {},
            })

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format"""
        lines = []
        with self.lock:
            for metric_name, samples in self.metrics.items():
                if not samples:
                    continue
                latest = samples[-1]
                labels_str = ", ".join(
                    f'{k}="{v}"' for k, v in latest.get("labels", {}).items()
                )
                labels_part = f"{{{labels_str}}}" if labels_str else ""
                lines.append(f"{metric_name}{labels_part} {latest['value']}")
        return "\n".join(lines)

    def get_stats(self, metric_name: str) -> Dict:
        """Get statistics for a metric"""
        with self.lock:
            samples = self.metrics.get(metric_name, [])
            if not samples:
                return {}
            values = [s["value"] for s in samples]
            return {
                "mean": np.mean(values),
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }


# ============================================================================
# ADVANCED FORECASTING (Prophet-like)
# ============================================================================

class AdvancedForecast(ABC):
    """Abstract forecaster for demand prediction"""

    @abstractmethod
    def update(self, demand: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def forecast(self, steps: int = 1) -> float:
        pass


class ExponentialSmoothingForecast(AdvancedForecast):
    """Holt-Winters triple exponential smoothing"""

    def __init__(self, alpha: float = 0.2, beta: float = 0.1, gamma: float = 0.1, season_length: int = 12):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.level = 0
        self.trend = 0
        self.seasonal = [1.0] * season_length
        self.history = []
        self.index = 0

    def update(self, demand: int) -> Tuple[float, float]:
        if not self.history:
            self.level = demand
            self.trend = 0
        else:
            prev_level = self.level
            # Deseasonalize
            deseasonalized = demand / self.seasonal[self.index % self.season_length]
            # Update level and trend
            self.level = self.alpha * deseasonalized + (1 - self.alpha) * (prev_level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
            # Update seasonal
            self.seasonal[self.index % self.season_length] = (
                self.gamma * (demand / self.level) +
                (1 - self.gamma) * self.seasonal[self.index % self.season_length]
            )
        self.history.append(demand)
        self.index += 1
        return self.level, self.trend

    def forecast(self, steps: int = 1) -> float:
        seasonal_idx = (self.index + steps - 1) % self.season_length
        return (self.level + steps * self.trend) * self.seasonal[seasonal_idx]


class LSTMForecast(AdvancedForecast):
    """LSTM-based forecasting (simplified without external library)"""

    def __init__(self, lookback: int = 12):
        self.lookback = lookback
        self.history = []
        # Simplified: use exponential weighted average for now
        self.ewma_short = None
        self.ewma_long = None

    def update(self, demand: int) -> Tuple[float, float]:
        self.history.append(demand)
        alpha_short = 0.3
        alpha_long = 0.1

        if self.ewma_short is None:
            self.ewma_short = demand
            self.ewma_long = demand
        else:
            self.ewma_short = alpha_short * demand + (1 - alpha_short) * self.ewma_short
            self.ewma_long = alpha_long * demand + (1 - alpha_long) * self.ewma_long

        trend = self.ewma_short - self.ewma_long
        return self.ewma_short, trend

    def forecast(self, steps: int = 1) -> float:
        trend = self.ewma_short - self.ewma_long
        return self.ewma_short + steps * trend


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

class HyperparameterTuner:
    """Grid/random search optimization for AP-CWS parameters"""

    def __init__(self, objective_fn: Callable, param_space: Dict[str, List], n_trials: int = 100):
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')
        self.history = []

    def random_search(self) -> Dict:
        """Random search optimization"""
        for trial in range(self.n_trials):
            params = {}
            for param_name, values in self.param_space.items():
                if isinstance(values, tuple) and len(values) == 2:
                    # Continuous range
                    params[param_name] = np.random.uniform(values[0], values[1])
                else:
                    # Discrete choices
                    params[param_name] = np.random.choice(values)

            score = self.objective_fn(params)
            self.history.append((params, score))

            if score < self.best_score:
                self.best_score = score
                self.best_params = params

        return self.best_params

    def grid_search(self) -> Dict:
        """Grid search optimization (for discrete params)"""
        import itertools

        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]

        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            score = self.objective_fn(params)
            self.history.append((params, score))

            if score < self.best_score:
                self.best_score = score
                self.best_params = params

        return self.best_params

    def bayesian_search(self, n_trials: int = 50) -> Dict:
        """Simplified Bayesian optimization using UCB strategy"""
        # Track param scores
        score_history = defaultdict(list)

        for trial in range(n_trials):
            best_ucb = float('-inf')
            best_params = None

            # Explore param combinations with highest upper confidence bound
            for _ in range(10):
                params = {}
                for param_name, values in self.param_space.items():
                    if isinstance(values, tuple):
                        params[param_name] = np.random.uniform(values[0], values[1])
                    else:
                        params[param_name] = np.random.choice(values)

                # UCB calculation (simplified)
                param_key = str(params)
                prev_scores = score_history.get(param_key, [])
                if prev_scores:
                    mean = np.mean(prev_scores)
                    ucb = mean - np.std(prev_scores) / np.sqrt(len(prev_scores))
                else:
                    ucb = 0

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_params = params

            score = self.objective_fn(best_params)
            score_history[str(best_params)].append(score)
            self.history.append((best_params, score))

            if score < self.best_score:
                self.best_score = score
                self.best_params = best_params

        return self.best_params


# ============================================================================
# COST-AWARE OPTIMIZATION
# ============================================================================

class CostOptimizer:
    """Cost-benefit analysis and optimization"""

    def __init__(self, nodes: List, lambda_cost: float = 0.1):
        self.nodes = nodes
        self.lambda_cost = lambda_cost  # Cost weight

    def compute_cost_function(
        self,
        cold_start_rate: float,
        idle_memory: float,
        workflow_latency: float,
        total_memory: float,
    ) -> float:
        """Multi-objective cost function to minimize"""
        # Weighted combination of objectives
        latency_cost = workflow_latency  # Want low latency
        cold_start_cost = 100.0 * cold_start_rate  # Want low cold starts
        idle_cost = self.lambda_cost * (idle_memory / total_memory)  # Want low idle

        return latency_cost + cold_start_cost + idle_cost

    def optimize_pool_allocation(
        self,
        functions: Dict,
        total_memory: int,
        target_cold_prob: float = 0.05,
    ) -> Dict[str, int]:
        """Allocate warm pools to minimize cost"""
        pool_sizes = {}
        remaining_memory = total_memory

        # Sort by criticality (high first)
        sorted_funcs = sorted(
            functions.items(),
            key=lambda x: x[1].get('criticality', 0),
            reverse=True
        )

        for func_id, func_data in sorted_funcs:
            memory = func_data.get('memory', 256)
            criticality = func_data.get('criticality', 0.5)

            # Allocate based on criticality and available memory
            allocation = min(
                int(criticality * 10),  # Up to 10 containers per critical function
                remaining_memory // memory
            )

            pool_sizes[func_id] = max(1, allocation)
            remaining_memory -= pool_sizes[func_id] * memory

        return pool_sizes


# ============================================================================
# ENHANCED AP-CWS WITH MONITORING & OPTIMIZATION
# ============================================================================

class EnhancedAPCWSScheduler:
    """AP-CWS with integrated monitoring, forecasting, and optimization"""

    def __init__(
        self,
        nodes: List,
        w_min: int = 1,
        w_max: int = 20,
        forecaster_type: str = "exponential",  # or "lstm"
        enable_optimization: bool = True,
    ):
        self.nodes = nodes
        self.w_min = w_min
        self.w_max = w_max
        self.forecaster_type = forecaster_type
        self.enable_optimization = enable_optimization

        # Metrics
        self.metrics = MetricsCollector()

        # Forecasters
        self.forecasters = {}
        self.forecaster_class = (
            ExponentialSmoothingForecast if forecaster_type == "exponential" else LSTMForecast
        )

        # Optimizers
        self.cost_optimizer = CostOptimizer(nodes, lambda_cost=0.1)
        self.hyperparams = {
            "alpha": 0.2,
            "beta": 0.1,
            "kappa": 0.5,
        }

        # State
        self.placement_history = []
        self.execution_log = []

    def get_forecaster(self, func_id: str) -> AdvancedForecast:
        """Get or create forecaster for function"""
        if func_id not in self.forecasters:
            self.forecasters[func_id] = self.forecaster_class()
        return self.forecasters[func_id]

    def optimize_hyperparameters(
        self,
        dag,
        invocation_history: Dict,
        num_trials: int = 20,
    ):
        """Tune α, β, κ using random search"""
        def objective(params):
            # Simulate scheduling with these params
            alpha, beta, kappa = params['alpha'], params['beta'], params['kappa']

            # Quick evaluation (simplified)
            forecaster = ExponentialSmoothingForecast(alpha=alpha, beta=beta)
            total_error = 0

            for func_id, history in invocation_history.items():
                for demand in history:
                    level, trend = forecaster.update(demand)
                    # Forecast error
                    if len(history) > 1:
                        total_error += abs(level - demand) ** 2

            return total_error

        param_space = {
            'alpha': (0.1, 0.5),
            'beta': (0.05, 0.2),
            'kappa': (0.3, 0.7),
        }

        tuner = HyperparameterTuner(objective, param_space, n_trials=num_trials)
        best_params = tuner.random_search()

        self.hyperparams.update(best_params)
        self.metrics.record_metric(
            "tuning_best_score",
            tuner.best_score,
            labels={"method": "random_search"}
        )

        return best_params

    def schedule_with_optimization(
        self,
        dag,
        invocation_history: Dict,
        slo_latency: float,
    ) -> Tuple[Dict, Dict, Dict]:
        """Execute planning with integrated optimization"""

        # 1. Optimize hyperparameters if enabled
        if self.enable_optimization:
            self.optimize_hyperparameters(dag, invocation_history, num_trials=20)

        # 2. Update forecasts
        for func_id, history in invocation_history.items():
            forecaster = self.get_forecaster(func_id)
            for demand in history:
                forecaster.update(demand)

        # 3. Prepare function criticality data
        functions_data = {}
        for func_id, func in dag.functions.items():
            functions_data[func_id] = {
                'memory': func.memory,
                'criticality': dag.criticality.get(func_id, 0.5),
            }

        # 4. Compute cost-optimal pool allocation
        total_memory = sum(node.total_memory for node in self.nodes)
        pool_sizes = self.cost_optimizer.optimize_pool_allocation(
            functions_data,
            total_memory,
            target_cold_prob=0.05,
        )

        # 5. Placement (simplified)
        placement = {}
        for func_id in dag.functions:
            # Assign to first available node
            for node in self.nodes:
                if node.free_memory >= dag.functions[func_id].memory:
                    placement[func_id] = node.id
                    node.free_memory -= dag.functions[func_id].memory
                    break

        # 6. Record metrics
        self.metrics.record_metric(
            "pool_sizes_avg",
            np.mean(list(pool_sizes.values())),
            labels={"dag": dag.id}
        )

        self.placement_history.append({
            "timestamp": datetime.now(),
            "placement": placement,
            "pool_sizes": pool_sizes,
            "hyperparams": self.hyperparams,
        })

        return pool_sizes, placement, self.hyperparams

    def export_metrics(self, output_file: str):
        """Export metrics to file"""
        with open(output_file, 'w') as f:
            f.write(self.metrics.export_prometheus())

    def get_summary(self) -> Dict:
        """Get scheduling summary"""
        return {
            "forecaster_type": self.forecaster_type,
            "hyperparameters": self.hyperparams,
            "metrics_snapshot": {
                name: self.metrics.get_stats(name)
                for name in ["pool_sizes_avg", "tuning_best_score"]
            },
            "placement_count": len(self.placement_history),
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from ap_cws_algorithm import (
        Node, Function, WorkflowDAG, APCWSScheduler
    )

    # Setup nodes
    nodes = [
        Node("node_1", total_memory=4096, speed=1.0, price_per_hour=0.05),
        Node("node_2", total_memory=4096, speed=1.2, price_per_hour=0.06),
        Node("node_3", total_memory=8192, speed=2.0, price_per_hour=0.10),
    ]

    # Create enhanced scheduler
    scheduler = EnhancedAPCWSScheduler(
        nodes,
        forecaster_type="exponential",
        enable_optimization=True,
    )

    # Setup workflow
    functions = {
        "f1": Function("f1", 256, 0.5, 0.5, [], ["f2", "f3"]),
        "f2": Function("f2", 512, 1.0, 1.0, ["f1"], ["f4"]),
        "f3": Function("f3", 256, 0.5, 0.3, ["f1"], ["f4"]),
        "f4": Function("f4", 1024, 2.0, 2.0, ["f2", "f3"], []),
    }
    dag = WorkflowDAG("workflow_1", functions, ["f1"], ["f4"])
    dag.criticality = {"f1": 0.8, "f2": 0.9, "f3": 0.3, "f4": 1.0}

    # Mock invocation history
    invocation_history = {
        "f1": [100, 110, 105, 115, 108, 120],
        "f2": [50, 55, 52, 60, 58, 65],
        "f3": [45, 50, 48, 55, 52, 60],
        "f4": [40, 42, 41, 45, 43, 48],
    }

    # Run optimization
    print("=== Enhanced AP-CWS with Optimization ===\n")
    pool_sizes, placement, hyperparams = scheduler.schedule_with_optimization(
        dag, invocation_history, slo_latency=5.0
    )

    print(f"Optimized Hyperparameters:\n{json.dumps(hyperparams, indent=2)}")
    print(f"\nPool Sizes:\n{json.dumps(pool_sizes, indent=2)}")
    print(f"\nPlacement:\n{json.dumps(placement, indent=2)}")

    # Export metrics
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "ap_cws_metrics.txt")
    scheduler.export_metrics(metrics_path)
    print(f"\nMetrics exported to {metrics_path}")

    # Summary
    summary = scheduler.get_summary()
    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
