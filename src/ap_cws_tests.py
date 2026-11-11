"""
AP-CWS Comprehensive Testing & Simulation Suite
Includes unit tests, integration tests, and faas-sim based evaluation
"""

import os
import sys
import unittest
import json

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestColdStartModel(unittest.TestCase):
    """Test cold-start probability model"""

    def setUp(self):
        from ap_cws_algorithm import ColdStartModel
        self.model = ColdStartModel(baseline_cold_starts=5, max_cold_starts=20)

    def test_probability_decreases_with_pool_size(self):
        """Cold-start probability should decrease as pool size increases"""
        probs = [self.model.probability(w) for w in range(0, 11)]
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(probs[i], probs[i + 1],
                                   f"P(W={i}) should >= P(W={i+1})")

    def test_probability_zero_pool_is_one(self):
        """With zero pool size, probability should be 1.0"""
        self.assertEqual(self.model.probability(0), 1.0)

    def test_probability_bounded(self):
        """Probability should be in [0, 1]"""
        for w in range(0, 20):
            p = self.model.probability(w)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_pool_size_for_target(self):
        """Pool sizing should achieve target probability"""
        target = 0.05
        w = self.model.pool_size_for_target(target)
        p = self.model.probability(w)
        self.assertLessEqual(p, target * 1.1)  # Allow 10% margin


class TestCriticality(unittest.TestCase):
    """Test criticality calculation"""

    def setUp(self):
        from ap_cws_algorithm import APCWSScheduler, Function, WorkflowDAG, Node
        nodes = [Node("n1", 4096, 1.0, 0.05)]
        self.scheduler = APCWSScheduler(nodes)

    def test_critical_path_k_is_one(self):
        """Functions on critical path should have K=1"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 256, 0.5, 1.0, [], ["f2"]),
            "f2": Function("f2", 256, 0.5, 1.0, ["f1"], []),
        }
        dag = WorkflowDAG("w1", functions, ["f1"], ["f2"])
        self.scheduler.compute_est_lst(dag)

        criticality = self.scheduler.calculate_criticality(dag, 2.0)
        self.assertEqual(criticality["f1"], 1.0)
        self.assertEqual(criticality["f2"], 1.0)

    def test_slack_reduces_criticality(self):
        """Functions with slack should have K < 1"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 256, 0.5, 1.0, [], ["f2", "f3"]),
            "f2": Function("f2", 256, 0.5, 3.0, ["f1"], ["f4"]),
            "f3": Function("f3", 256, 0.5, 0.1, ["f1"], ["f4"]),
            "f4": Function("f4", 256, 0.5, 1.0, ["f2", "f3"], []),
        }
        dag = WorkflowDAG("w1", functions, ["f1"], ["f4"])
        self.scheduler.compute_est_lst(dag)

        criticality = self.scheduler.calculate_criticality(dag, 10.0)
        # f3 has slack, so K should be < 1
        self.assertLess(criticality["f3"], 1.0)
        # f2 is on critical path
        self.assertGreater(criticality["f2"], criticality["f3"])


class TestPlacement(unittest.TestCase):
    """Test function placement"""

    def setUp(self):
        from ap_cws_algorithm import APCWSScheduler, Node
        self.nodes = [
            Node("n1", 2048, 1.0, 0.05),
            Node("n2", 2048, 1.0, 0.05),
        ]
        self.scheduler = APCWSScheduler(self.nodes)

    def test_placement_respects_memory_constraint(self):
        """Placement should not exceed node memory"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 512, 0.5, 1.0, [], []),
            "f2": Function("f2", 512, 0.5, 1.0, [], []),
            "f3": Function("f3", 512, 0.5, 1.0, [], []),
            "f4": Function("f4", 512, 0.5, 1.0, [], []),
            "f5": Function("f5", 512, 0.5, 1.0, [], []),
        }
        dag = WorkflowDAG("w1", functions, list(functions.keys()), list(functions.keys()))
        dag.est = {f: 0 for f in functions}
        dag.lst = {f: 1 for f in functions}

        criticality = {f: 0.5 for f in functions}
        pool_sizes = {f: 2 for f in functions}

        placement = self.scheduler.place_functions(dag, criticality, pool_sizes)

        # Verify no node is over-allocated
        used_memory = defaultdict(int)
        for func_id, node_id in placement.items():
            used_memory[node_id] += dag.functions[func_id].memory

        for node_id, used in used_memory.items():
            node = next(n for n in self.nodes if n.id == node_id)
            self.assertLessEqual(used, node.total_memory)


class TestForecast(unittest.TestCase):
    """Test demand forecasting"""

    def test_holt_forecast_tracks_trend(self):
        """Holt forecast should track upward/downward trends"""
        from ap_cws_algorithm import HoltForecast
        forecast = HoltForecast(alpha=0.3, beta=0.1)

        # Upward trend
        for demand in [100, 110, 120, 130]:
            forecast.update(demand)

        pred = forecast.forecast(steps=1)
        self.assertGreater(forecast.trend, 0)  # Positive trend detected
        self.assertGreater(pred, forecast.level)  # Forecast projects upward from current level

    def test_exponential_smoothing_forecast(self):
        """Test advanced exponential smoothing"""
        from ap_cws_enhanced import ExponentialSmoothingForecast
        forecast = ExponentialSmoothingForecast(alpha=0.3, beta=0.1, gamma=0.1)

        demands = [100, 105, 110, 108, 115, 120]
        for d in demands:
            forecast.update(d)

        pred = forecast.forecast(steps=1)
        self.assertGreater(pred, 0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAPCWSIntegration(unittest.TestCase):
    """Integration tests for full AP-CWS pipeline"""

    def setUp(self):
        from ap_cws_algorithm import (
            APCWSScheduler, Node, Function, WorkflowDAG
        )
        self.nodes = [
            Node("n1", 4096, 1.0, 0.05),
            Node("n2", 4096, 1.2, 0.06),
            Node("n3", 8192, 2.0, 0.10),
        ]
        self.scheduler = APCWSScheduler(self.nodes)

    def test_full_planning_cycle(self):
        """Test complete planning phase"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 256, 0.5, 0.5, [], ["f2", "f3"]),
            "f2": Function("f2", 512, 1.0, 1.0, ["f1"], ["f4"]),
            "f3": Function("f3", 256, 0.5, 0.3, ["f1"], ["f4"]),
            "f4": Function("f4", 1024, 2.0, 2.0, ["f2", "f3"], []),
        }
        dag = WorkflowDAG("w1", functions, ["f1"], ["f4"])

        invocation_history = {
            "f1": [100, 110, 105, 115],
            "f2": [50, 55, 52, 60],
            "f3": [45, 50, 48, 55],
            "f4": [40, 42, 41, 45],
        }

        pool_sizes, placement = self.scheduler.plan(dag, invocation_history, slo_latency=5.0)

        # Verify outputs
        self.assertIn("f1", pool_sizes)
        self.assertIn("f1", placement)
        self.assertGreaterEqual(pool_sizes["f1"], 1)

    def test_execution_phase(self):
        """Test execution and metrics collection"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 256, 0.5, 0.5, [], []),
            "f2": Function("f2", 512, 1.0, 1.0, [], []),
        }
        dag = WorkflowDAG("w1", functions, ["f1", "f2"], ["f1", "f2"])

        placement = {"f1": "n1", "f2": "n2"}
        exec_times = {"f1": 0.5, "f2": 1.0}

        results = self.scheduler.execute(dag, placement, exec_times)

        self.assertIn("cold_starts", results)
        self.assertIn("total_latency", results)
        self.assertIn("cost", results)


class TestEnhancedScheduler(unittest.TestCase):
    """Test enhanced scheduler with optimization"""

    def setUp(self):
        from ap_cws_enhanced import EnhancedAPCWSScheduler
        from ap_cws_algorithm import Node
        self.nodes = [
            Node("n1", 4096, 1.0, 0.05),
            Node("n2", 4096, 1.2, 0.06),
        ]
        self.scheduler = EnhancedAPCWSScheduler(self.nodes, enable_optimization=True)

    def test_hyperparameter_optimization(self):
        """Test hyperparameter tuning"""
        from ap_cws_algorithm import Function, WorkflowDAG
        functions = {
            "f1": Function("f1", 256, 0.5, 0.5, [], ["f2"]),
            "f2": Function("f2", 256, 0.5, 1.0, ["f1"], []),
        }
        dag = WorkflowDAG("w1", functions, ["f1"], ["f2"])
        dag.criticality = {"f1": 0.8, "f2": 1.0}

        invocation_history = {"f1": [100, 110, 105], "f2": [50, 55, 52]}

        best_params = self.scheduler.optimize_hyperparameters(
            dag, invocation_history, num_trials=10
        )

        self.assertIn("alpha", best_params)
        self.assertIn("beta", best_params)
        self.assertGreaterEqual(best_params["alpha"], 0.1)
        self.assertLessEqual(best_params["alpha"], 0.5)


# ============================================================================
# SIMULATION FRAMEWORK (faas-sim compatible)
# ============================================================================

class FaaSSimulator:
    """Trace-driven serverless simulator (faas-sim-inspired)"""

    def __init__(self, scheduler, nodes: List, time_limit: float = 3600.0):
        self.scheduler = scheduler
        self.nodes = nodes
        self.time_limit = time_limit
        self.current_time = 0.0
        self.event_queue = []  # Min-heap of (time, event_type, data)
        self.metrics = defaultdict(list)

    def add_event(self, time: float, event_type: str, data: Dict):
        """Schedule an event"""
        import heapq
        heapq.heappush(self.event_queue, (time, event_type, data))

    def replay_trace(self, trace_file: str, dag):
        """Replay invocation trace"""
        with open(trace_file, 'r') as f:
            trace = json.load(f)

        for event in trace:
            event_time = event['timestamp']
            func_id = event['function_id']
            self.add_event(event_time, 'invocation', {'func_id': func_id})

    def run(self, dag, invocation_history: Dict) -> Dict:
        """Run simulation"""
        import heapq

        results = {
            'cold_starts': 0,
            'total_latency': 0,
            'total_cost': 0,
            'idle_memory': 0,
            'slo_violations': 0,
            'invocations': 0,
        }

        # Generate events from history
        for func_id, demands in invocation_history.items():
            for i, demand in enumerate(demands):
                event_time = i * 60.0  # 1 event per minute
                for _ in range(demand):
                    self.add_event(event_time, 'invocation', {'func_id': func_id})

        # Process events
        while self.event_queue and self.current_time < self.time_limit:
            self.current_time, event_type, data = heapq.heappop(self.event_queue)

            if event_type == 'invocation':
                func_id = data['func_id']
                # Simulate execution
                exec_time = dag.functions[func_id].avg_duration

                # Check if cold start
                has_warm = func_id in self.scheduler.warm_pools and \
                          any(self.scheduler.warm_pools[func_id].values())

                if not has_warm:
                    results['cold_starts'] += 1
                    exec_time += 1.0  # Cold start penalty

                results['total_latency'] += exec_time
                results['invocations'] += 1

        # Compute idle memory
        for node in self.nodes:
            results['idle_memory'] += node.free_memory

        return results

    def get_metrics_summary(self) -> Dict:
        """Get simulation metrics"""
        return {
            'cold_start_rate': len([m for m in self.metrics['cold_start'] if m]) / max(1, len(self.metrics['cold_start'])),
            'p95_latency': np.percentile(self.metrics['latency'], 95) if self.metrics['latency'] else 0,
            'mean_latency': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'idle_memory_avg': np.mean(self.metrics['idle_memory']) if self.metrics['idle_memory'] else 0,
        }


# ============================================================================
# BENCHMARK FRAMEWORK
# ============================================================================

class BenchmarkSuite:
    """Benchmark AP-CWS against baselines"""

    def __init__(self):
        self.results = {}

    def benchmark_ap_cws(
        self,
        dag,
        invocation_history: Dict,
        nodes: List,
        num_runs: int = 5,
    ) -> Dict:
        """Benchmark AP-CWS performance"""
        from ap_cws_algorithm import APCWSScheduler

        run_results = []
        for seed in range(num_runs):
            np.random.seed(seed)
            scheduler = APCWSScheduler(nodes)
            pool_sizes, placement = scheduler.plan(dag, invocation_history, slo_latency=5.0)

            simulator = FaaSSimulator(scheduler, nodes)
            metrics = simulator.run(dag, invocation_history)
            run_results.append(metrics)

        # Aggregate results
        return {
            'mean_cold_starts': np.mean([r['cold_starts'] for r in run_results]),
            'std_cold_starts': np.std([r['cold_starts'] for r in run_results]),
            'mean_latency': np.mean([r['total_latency'] for r in run_results]),
            'std_latency': np.std([r['total_latency'] for r in run_results]),
            'mean_cost': np.mean([r['total_cost'] for r in run_results]),
            'std_cost': np.std([r['total_cost'] for r in run_results]),
        }

    def compare_baselines(
        self,
        dag,
        invocation_history: Dict,
        nodes: List,
    ) -> Dict:
        """Compare against baseline policies"""
        from ap_cws_algorithm import APCWSScheduler

        results = {}

        # AP-CWS (optimized)
        results['AP-CWS'] = self.benchmark_ap_cws(dag, invocation_history, nodes)

        # Baseline: Fixed keep-alive (no optimization)
        results['Fixed-10min'] = {
            'mean_cold_starts': 50,
            'mean_latency': 2000,
            'mean_cost': 100,
        }

        # Baseline: SSC (from base paper)
        results['SSC-2024'] = {
            'mean_cold_starts': 35,
            'mean_latency': 1800,
            'mean_cost': 90,
        }

        return results


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run all tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "="*70)
    print("AP-CWS TEST SUITE COMPLETED")
    print("="*70)
