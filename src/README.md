# AP-CWS: Predictive Cold-Start-Aware Serverless Workflow Scheduler
## Source Code & Simulation Execution Guide

**Institution:** Indian Institute of Technology (BHU) Varanasi  
**Base Research Paper:** Pan et al., *"Sustainable Serverless Computing With Cold-Start Optimization and Automatic Workflow Resource Scheduling"*, IEEE Transactions on Sustainable Computing (IEEE T-SUSC), 2024.  

**Authors:**
- **Nikhil Sharma**
- **Raghav Arora**
- **Jashanpreet Singh**

---

## 1. Overview & Core Mechanisms

Serverless function workflows (FaaS DAGs) suffer from severe tail latency penalties caused by **cascading cold starts**: when a task on the critical execution path experiences a cold initialization delay, every subsequent dependent task is blocked, multiplying total pipeline latency and causing service-level objective (SLO) violations.

Existing state-of-the-art approaches like **SSC (Pan et al., IEEE T-SUSC 2024)** size container pools purely based on aggregate invocation frequency, ignoring workflow topology and critical path slack. **AP-CWS** solves this through a dual-loop predictive architecture:

```text
+-----------------------------------------------------------------------------------------+
|                                    AP-CWS ARCHITECTURE                                  |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [PLANNING LOOP - Periodic Epoch (Δ = 60s)]                                             |
|   1. Demand Forecasting   : Holt Linear Exponential Smoothing (alpha=0.3, beta=0.1)     |
|   2. DAG Criticality (K_i): Topological slack index K_i = 1 - Slack_i / Makespan [0, 1] |
|   3. Target Cold Budget   : eps_i = eps_max - (eps_max - eps_min) * K_i (0.25 -> 0.02)  |
|   4. Warm Pool Sizing     : Analytical warm pool sizing W_i per function across nodes   |
|                                                                                         |
|  [DISPATCH LOOP - Event-Driven Invocations (<50 us)]                                    |
|   1. Joint Placement Score: S_ij = (1 + eta*K_i)*(y_ij*s_i + q_j + t_exec) + mu*c_ij    |
|   2. JIT Prewarming       : Prewarms critical successors (K_child >= kappa = 0.60)      |
|   3. Container Lifecycle  : SimPy event-driven Uninstantiated -> Warm -> Busy -> Evict  |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

### Key Mathematical Formulations (`APCWSSchedulerPolicy`):
1. **Holt Linear Exponential Smoothing Demand Forecast:**
   $$\ell_t = \alpha D_{i, t} + (1 - \alpha)(\ell_{t-1} + b_{t-1}), \qquad b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$

2. **DAG Topological Criticality ($K_i \in [0, 1]$):**
   $$K_i = 1 - \frac{\text{Slack}_i}{T^*}, \qquad \text{Slack}_i = LST_i - EST_i$$
   Critical-path functions have zero slack ($K_i = 1.0$).

3. **Criticality-Aware Cold-Start Budget:**
   $$\varepsilon_i = \varepsilon_{\max} - (\varepsilon_{\max} - \varepsilon_{\min}) \cdot K_i$$
   Default parameters in [`APCWSSchedulerPolicy`](ap_cws_baselines.py): $\varepsilon_{\min} = 0.02$ (critical path), $\varepsilon_{\max} = 0.25$ (high-slack branches).

4. **Joint Placement Scoring Function:**
   $$S_{ij} = (1 + \eta K_i)\left(y_{ij} s_i + q_j + \frac{t_i^{\text{base}}}{v_j}\right) + \mu c_{ij}$$
   where $\eta = 1.5$, $\mu = 0.05$, and $y_{ij} \in \{0, 1\}$ indicates cold-start penalty ($0$ if idle warm container exists on node $j$).

5. **Just-In-Time (JIT) Successor Prewarming:**
   When a task begins on a node, any child task with $K_{\text{child}} \ge \kappa$ ($\kappa = 0.60$) lacking an idle warm instance is prewarmed concurrently.

---

## 2. Directory Structure & File Manifest

All executable simulation code, baseline implementations, unit tests, and benchmark CSV datasets reside inside `src/`:

```text
src/
├── README.md                      # Module documentation & execution instructions
├── demo.py                        # Interactive CLI defense suite (supports CLI args & menu)
├── ap_cws_simpy_engine.py         # Discrete-event SimPy cluster simulator (paper experiments)
├── ap_cws_baselines.py            # Policies: AP-CWS, SSC 2024, CASR 2025, MPC 2025, DROPS 2026 & ablations
├── run_2nd_evaluation.py          # 324-run empirical benchmark harness across 3 DAGs & 4 traces
├── run_scalability_stress_test.py # Stress sweep (1-32 req/s) & Fast-AP-CWS scaling (16-256 nodes)
├── ap_cws_algorithm.py            # Standalone Phase-1 prototype scheduler
├── ap_cws_enhanced.py             # Auxiliary monitoring & hyperparameter-tuning extensions
├── ap_cws_tests.py                # 12-test unit & integration suite for the core/enhanced prototype
└── results/                       # Empirical CSV datasets from evaluations
    ├── simulation_results_raw.csv         # 324 individual simulation run records
    ├── simulation_results_summary.csv     # Aggregated P95, mean latency, cold starts, cost
    ├── scalability_stress_results.csv     # Throughput sweep data (breaking point analysis)
    └── scalability_overhead_results.csv   # O(1) vs O(|J|) scheduler overhead scaling data
```

---

## 3. Environment Setup & Prerequisites

### Hardware & Software Requirements:
- **Operating System:** Platform-agnostic (Windows 10/11, Ubuntu 20.04/22.04/24.04, macOS)
- **Python Version:** Python 3.8+ (tested through Python 3.14)
- **Dependencies:** `simpy`, `numpy`, `pandas`, `matplotlib`, `scipy` (listed in root [`../requirements.txt`](../requirements.txt))
- **External Dependencies:** None (no Docker daemon, cloud credentials, or external database required).

### 1-Line Installation Command:
From the repository root directory:

```bash
pip install -r requirements.txt
```

---

## 4. HOW TO RUN: Step-by-Step Execution Guide

### Option A: Interactive Defense CLI Suite
From the repository root directory, run:

```bash
python src/demo.py
```

This launches the terminal menu:
```text
========================================================================================
  Predictive, Cold-Start-Aware Serverless Workflow Scheduler
  IIT (BHU) Varanasi -- Cloud Computing (2nd Evaluation Oral Defense)
========================================================================================
SELECT DEMONSTRATION MODULE:
  [1] Quick Verification Smoke Test        (~3s : 2 nodes, live event trace, SimPy engine check)
  [2] Comparative Benchmark Run            (~10s: AP-CWS vs SSC 2024 vs DROPS 2026 vs Fixed Keep-Alive)
  [3] Hyperscale Stress & Breaking Point   (~10s: Throughput sweep 1-25 req/s, 12 req/s cliff vs 22 req/s RAM limit)
  [4] Computational Overhead & Fast-AP-CWS (~5s : 10,000 dispatches on 128 nodes, O(1) vs O(N) 3x speedup)
  [5] Display Summary Evaluation Scorecard (Full 324-run empirical benchmark summary directly on screen)
  [6] Exit CLI Demonstration
```

---

### Option B: Headless / Scripted Direct Execution Commands

Each demonstration module can also be launched directly from the repository root without interactive prompts:

#### 1. Quick Verification Smoke Test (~3s)
Verifies the SimPy discrete-event engine, container state transitions (Alloc -> Warm -> Busy -> Idle), and JIT prewarm triggers on a 2-node cluster with live timestamped event logging:
```bash
python src/demo.py 1
# OR:
python src/demo.py smoke
```

#### 2. Comparative Benchmark Run (~10s)
Executes a live head-to-head comparison between **AP-CWS**, **SSC (Pan et al., 2024)**, **DROPS (2026)**, and **Fixed 10-min Keep-Alive** on an 8-function Video Analytics workflow:
```bash
python src/demo.py 2
# OR:
python src/demo.py bench
```

#### 3. Hyperscale Stress & Breaking Point (~10s)
Executes an arrival rate sweep from 1 to 25 requests/second, reproducing the **12 req/s Cascading Latency Cliff** where queueing backpressure collides with cold starts:
```bash
python src/demo.py 3
# OR:
python src/demo.py stress
```

#### 4. Computational Overhead & Fast-AP-CWS (~5s)
Executes 10,000 function dispatches on a 128-node cluster, showing `FastAPCWSSchedulerPolicy` ($O(1)$ sampled selection + memoized criticality) achieves a **~3.0x speedup** over $O(|\mathcal{J}|)$ scanning (>71,000 decisions/second):
```bash
python src/demo.py 4
# OR:
python src/demo.py overhead
```

#### 5. Full Evaluation Scorecard (Instant)
Displays the summary evaluation scorecard across the 324-run simulation benchmark and ablation variants:
```bash
python src/demo.py 5
# OR:
python src/demo.py scorecard
```

#### 6. Run All Demo Modules in Sequence
```bash
python src/demo.py all
```

---

### Option C: Run the Unit & Integration Test Suite

Validates mathematical invariants, Holt forecasting trend tracking, criticality boundaries, pool sizing bounds, and node memory capacity constraints across 12 automated tests for `ap_cws_algorithm.py` and `ap_cws_enhanced.py`:

```bash
python src/ap_cws_tests.py
# OR via unittest discovery from repository root:
python -m unittest discover -s src -p "*tests.py" -v
```

**Expected Output:**
```text
test_execution_phase (__main__.TestAPCWSIntegration.test_execution_phase) ... ok
test_full_planning_cycle (__main__.TestAPCWSIntegration.test_full_planning_cycle) ... ok
test_pool_size_for_target (__main__.TestColdStartModel.test_pool_size_for_target) ... ok
test_probability_bounded (__main__.TestColdStartModel.test_probability_bounded) ... ok
test_probability_decreases_with_pool_size (__main__.TestColdStartModel.test_probability_decreases_with_pool_size) ... ok
test_probability_zero_pool_is_one (__main__.TestColdStartModel.test_probability_zero_pool_is_one) ... ok
test_critical_path_k_is_one (__main__.TestCriticality.test_critical_path_k_is_one) ... ok
test_slack_reduces_criticality (__main__.TestCriticality.test_slack_reduces_criticality) ... ok
test_hyperparameter_optimization (__main__.TestEnhancedScheduler.test_hyperparameter_optimization) ... ok
test_exponential_smoothing_forecast (__main__.TestForecast.test_exponential_smoothing_forecast) ... ok
test_holt_forecast_tracks_trend (__main__.TestForecast.test_holt_forecast_tracks_trend) ... ok
test_placement_respects_memory_constraint (__main__.TestPlacement.test_placement_respects_memory_constraint) ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.07s

OK
======================================================================
AP-CWS TEST SUITE COMPLETED
======================================================================
```

---

### Option D: Re-run Full 324-Run Empirical Evaluation Harness

To re-execute the exhaustive simulation across all 324 experimental runs (3 workflow DAGs $\times$ 4 arrival patterns $\times$ 9 policies including ablations $\times$ 3 seeds):

```bash
python src/run_2nd_evaluation.py
```

*Note: Writes datasets to `src/results/simulation_results_raw.csv` and `src/results/simulation_results_summary.csv`, and saves charts `fig1`–`fig6` into `docs/figures/`.*

---

### Option E: Re-run Scalability Stress & Overhead Suite

To re-run the 1–32 req/s breaking-point throughput sweep and the 16–256 node scheduler overhead benchmark:

```bash
python src/run_scalability_stress_test.py
```

*Note: Writes datasets to `src/results/scalability_stress_results.csv` and `src/results/scalability_overhead_results.csv`, and saves `fig7`–`fig8` into `docs/figures/`.*

---

## 5. Summary Scorecard & Empirical Comparison

Results aggregated across **324 simulation runs** (from [`results/simulation_results_raw.csv`](results/simulation_results_raw.csv) and [`results/scalability_overhead_results.csv`](results/scalability_overhead_results.csv)):

| Metric | AP-CWS (Proposed) | SSC (Pan 2024) | DROPS (EuroSys 2026) | Fixed Keep-Alive (10m) | AP-CWS vs SSC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P95 Tail Latency** | **2.83 s** | 3.92 s | 5.18 s | 4.68 s | **27.8% lower** |
| **Mean End-to-End Latency** | **2.41 s** | 2.83 s | 3.68 s | 3.37 s | **14.8% lower** |
| **Overall Cold-Start Rate** | **0.58%** | 2.26% | 2.29% | 2.23% | **74.3% reduction** |
| **Critical-Path Cold Starts** | **0.63%** | 2.34% | 2.35% | 2.29% | **73.1% reduction** |
| **SLO Violation Rate** | **0.46%** | 3.16% | 4.51% | 3.70% | **85.4% reduction** |
| **5-Min Execution + Idle Cost** | **\$0.0390** | \$0.0409 | \$0.0466 | \$0.0344 | **4.6% savings** |
| **Dispatch Overhead (128 Nodes)** | **13.94 $\mu$s** *(Fast-AP-CWS)* | 4.87 $\mu$s | 15.31 $\mu$s | — | **2.97x speedup** *(vs 41.43 $\mu$s AP-CWS)* |

### Ablation Study Summary (Mean Across All 324 Runs):
1. **Without JIT Prewarming (`APCWSAblationNoPrewarm`, $\kappa = \infty$):** P95 tail latency degrades by **+25.8%** (from 2.83 s to 3.56 s), critical-path cold starts rise from 0.63% to 2.27%, and SLO violations surge to 3.02% (in Montage Scientific, P95 degrades by +36.0% from 4.84 s to 6.58 s).
2. **Without DAG Criticality (`APCWSAblationNoCriticality`, $K_i = 0$):** Static warm pools absorb steady traffic, while uniform slack budgets leave critical joins unprotected during spikes.
3. **Without Cost Weight (`APCWSAblationNoCost`, $\mu = 0$):** Placement prioritizes latency and RAM headroom only, ignoring node price differences.

---

## 6. Design & Defense Q&A Quick Reference

- **Q: How does AP-CWS differ from Pan et al. (SSC 2024)?**  
  *A:* SSC decouples frequency-based pool sizing from task placement without topological slack budgeting or locality-aware scoring. AP-CWS couples cold-start probability budgeting ($\varepsilon_i = \varepsilon_{\max} - (\varepsilon_{\max} - \varepsilon_{\min}) K_i$), multi-factor placement scoring ($S_{ij}$), and JIT successor prewarming, cutting P95 latency by 27.8% (2.83 s vs 3.92 s), SLO violations from 3.16% down to 0.46%, and total 5-minute cost by 4.6% ($0.0390 vs $0.0409).
- **Q: How is the simulation validated?**  
  *A:* [`ap_cws_simpy_engine.py`](ap_cws_simpy_engine.py) implements a `SimPy` discrete-event cluster simulator modeling container cold starts, heterogeneous node speeds, execution concurrency, network transfers, idle keep-alive timeouts, evictions, and time-integral memory & cost accounting.
- **Q: Where does the system break down under heavy load?**  
  *A:* At $\lambda = 12$ req/s, decoupled baselines hit the **Cascading Latency Cliff** (P95 spikes to 14.39 s under SSC and 14.88 s under DROPS/Fixed Keep-Alive), whereas AP-CWS holds P95 at 2.54 s until reaching the physical 64 GB RAM limit at $\lambda = 22$ req/s.

---
*For research papers, architectural diagrams, evaluation plots, and slide decks, refer to [`../paper/`](../paper) and [`../docs/`](../docs).*
