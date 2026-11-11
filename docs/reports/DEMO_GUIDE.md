# AP-CWS: 2nd Evaluation Oral Defense & Live Demonstration Guide

**Project Title:** AP-CWS: Predictive Cold-Start-Aware Serverless Workflow Scheduler  
**Course:** Cloud Computing  
**Institution:** Indian Institute of Technology (BHU) Varanasi  
**Evaluation Stage:** 2nd Evaluation (Weightage: 20%)  
**Team Members:**  
- **Nikhil Sharma** (Roll No: 24135083)  
- **Raghav Arora** (Roll No: 24135097)  
- **Jashanpreet Singh** (Roll No: 24134015)  
**Base Research Paper:** Pan et al., *"Sustainable Serverless Computing With Cold-Start Optimization and Automatic Workflow Resource Scheduling"*, IEEE Transactions on Sustainable Computing (T-SUSC), 2024.

---

## 1. Quick Launch Instructions

### Prerequisites
- Python 3.8+ (tested through Python 3.14)
- Installed dependencies (`pip install -r requirements.txt` from repo root): `simpy`, `numpy`, `pandas`, `matplotlib`, `scipy`

### Launch Command
Open your terminal (PowerShell, Command Prompt, or bash) in the repository root directory and run:

```bash
python src/demo.py
```

The interactive terminal banner will appear with the 6-module menu.

---

## 2. Interactive Menu Overview

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

## 3. Step-by-Step Demonstration Cheatsheet & 30-Second Speaking Scripts

### Option 1: Quick Verification Smoke Test
- **Command to type:** `1` (then press Enter)
- **Duration:** ~3 seconds
- **What it does:** Spawns a 2-node cluster and executes the 6-function E-Commerce DAG across 4 successive workflow requests, streaming discrete-event logs in real-time.

#### 30-Second Speaking Script:
> *"Respected Professor, this first test verifies the operational integrity of our discrete-event simulation engine built with SimPy. As you see on screen, when the first workflow arrives, the initial task experiences a 0.6-second cold start. But immediately, AP-CWS calculates the DAG topological criticality and detects that child tasks 'fraud' and 'payment' are on the critical path with $K_i = 1.0$. It triggers Just-In-Time proactive prewarming. Notice how subsequent tasks and later workflow instances achieve instantaneous 0.00-second warm hits, proving that our container lifecycle state machine and prewarming mechanisms function flawlessly."*

#### Numbers to Highlight to the Professor:
- **Warm Reuse Rate:** `91.7%` warm hits achieved on steady requests.
- **Topological Criticality ($K_i$):** Critical path (`validate` $\to$ `fraud` $\to$ `payment` $\to$ `notify`) assigned $K_i = 1.00$, while slack tasks (`inventory`, `credit`) receive $K_i \approx 0.62$.
- **Engine Verification:** SimPy discrete-event clock, container allocation, and memory integrals verified `[PASS]`.

---

### Option 2: Comparative Benchmark Run
- **Command to type:** `2` (then press Enter)
- **Duration:** ~10 seconds
- **What it does:** Runs AP-CWS head-to-head against the Base Paper (SSC Pan 2024), EuroSys 2026 (DROPS), and Industry Fixed Keep-Alive under real-world Azure bursty traffic on the 8-function Video Analytics DAG.

#### 30-Second Speaking Script:
> *"Here we run a live comparative benchmark on the Video Analytics pipeline under real-world Azure bursty traffic. As the benchmark concludes, observe the comparative table: AP-CWS achieves a P95 tail latency of 3.36 seconds, compared to 6.14 seconds for the base paper SSC and 5.28 seconds for DROPS. That is an empirical 45% reduction in tail latency. More importantly, look at critical-path cold starts: AP-CWS slashes them to 1.8% compared to 7.1% in SSC, which directly translates to zero SLO violations during this burst window."*

#### Numbers to Highlight to the Professor:
- **P95 Latency:** `3.36s` (AP-CWS) vs `6.14s` (SSC 2024) $\to$ **~45% lower tail latency**.
- **Critical-Path Cold Starts:** `1.8%` vs `7.1%` in SSC $\to$ **~74% reduction**.
- **SLO Violations:** `0.0%` for AP-CWS vs `7.9%` in SSC and `9.2%` in DROPS.
- **Why SSC Fails:** SSC prioritizes tasks purely by execution duration and places them on the fastest node even if that node is cold. AP-CWS incorporates warm container presence ($y_{ij} s_i$) directly into the joint placement score $S_{ij}$.

---

### Option 3: Hyperscale Stress Test & Breaking Point Demo
- **Command to type:** `3` (then press Enter)
- **Duration:** ~10 seconds
- **What it does:** Sweeps arrival throughput from 1.0 to 25.0 req/s across an 8-node (64 GB RAM) cluster, revealing the exact system breaking point.

#### 30-Second Speaking Script:
> *"This stress test demonstrates the architectural resilience and breaking point of the cluster under increasing concurrency. From 1 to 5 req/s, all systems operate normally. However, at 12 req/s, the baselines hit what we call the 'Latency Cliff'—their P95 latency collapses from 2.4 seconds to over 15 seconds because uncoordinated cold starts create CPU and container initialization lockups. In contrast, AP-CWS holds rock-steady at 2.48 seconds! AP-CWS maintains stability until 22 to 25 req/s, where Little's Law dictates that physical RAM is 100% saturated. This proves AP-CWS pushes cluster efficiency to the actual physical hardware limit."*

#### Numbers to Highlight to the Professor:
- **The 12 req/s "Latency Cliff":** Baselines jump from 2.4s to **15.5s** (SLO collapse). AP-CWS stays at **2.48s** (6x better latency).
- **Breaking Point Distinction:**
  - *Baselines break at 12 req/s* due to **algorithmic lockup** (cold start storm).
  - *AP-CWS holds steady until 22-25 req/s*, reaching the **physical Little's Law capacity limit** ($N = \lambda W$, where 200 tasks/s $\times$ 1.5s duration = 300 concurrent containers $> 64$ GB).

---

### Option 4: Computational Overhead & Fast-AP-CWS Demo
- **Command to type:** `4` (then press Enter)
- **Duration:** ~5 seconds
- **What it does:** Benchmarks 10,000 consecutive task dispatches on a large 128-node cluster, measuring microsecond-level decision latency of Fast-AP-CWS vs Standard AP-CWS and baselines.

#### 30-Second Speaking Script:
> *"A critical question for production serverless schedulers is computational dispatch overhead. On a large cluster of 128 nodes, a naive full scan takes $O(|J|)$ time. In Option 4, we evaluate our production enhancement: Fast-AP-CWS. By applying static DAG criticality memoization and the mathematical Power-of-Two-Choices sub-sampling theorem from Azar et al., Fast-AP-CWS reduces decision overhead from 40.5 microseconds down to just 13.0 microseconds. That is a 3.1x speedup, enabling a single scheduler thread to dispatch over 77,000 serverless functions per second with negligible loss in scheduling quality."*

#### Numbers to Highlight to the Professor:
- **Dispatch Latency:** `12.98 us/decision` (Fast-AP-CWS) vs `40.49 us` (Standard AP-CWS).
- **Speedup:** `3.1x faster` ($O(1)$ vs $O(|J|)$ scaling).
- **Throughput:** `>77,000 dispatches/sec` per CPU core.
- **Theoretical Basis:** Azar et al. Power-of-Two-Choices theorem ($d=2$) proves maximum queue length is bounded to $O(\log \log n / \log 2)$ without scanning all $n$ nodes.

---

### Option 5: Display Summary Evaluation Scorecard
- **Command to type:** `5` (then press Enter)
- **Duration:** Instantaneous (~1 second)
- **What it does:** Displays the complete 324-run empirical benchmark scorecard across all 3 workflow topologies (Video Analytics, Montage, E-Commerce) and 4 Azure arrival profiles, plus the component ablation study.

#### 30-Second Speaking Script:
> *"Finally, Option 5 summarizes our complete experimental evaluation comprising 324 full simulation runs across 3 workflow DAGs, 4 real-world arrival patterns, and 3 randomized seeds. Across all topologies, AP-CWS delivers a mean P95 latency of 2.83 seconds versus 3.92 seconds for SSC 2024—a 27.8% improvement. Critical-path cold starts are virtually eliminated down to 0.6%, and SLO violations decrease by 84.4%. Our ablation study confirms that Just-In-Time prewarming contributes a 36% reduction in P95 latency, while topological criticality prevents slack tasks from wasting cluster memory."*

#### Numbers to Highlight to the Professor:
- **Overall P95 Latency:** `2.83s` (AP-CWS) vs `3.92s` (SSC 2024) $\to$ **27.8% improvement**.
- **SLO Violation Rate:** `0.5%` (AP-CWS) vs `3.2%` (SSC 2024) $\to$ **84.4% fewer violations**.
- **Critical-Path Cold Starts:** Slashed from `2.3%` to `0.6%`.
- **Ablation Study (Table 3):** Removing JIT Prewarming causes P95 latency to degrade from `4.84s` to `6.58s` (+36% penalty).

---

## 4. Anticipated Professor Questions & Model Answers

### Q1: *"Why does SSC (Pan 2024) have higher cold starts if it also prewarms?"*
**Answer:**  
SSC prewarms based strictly on invocation volume trends ($\Delta \lambda / \Delta t$) without structural awareness of the DAG critical path. Consequently, it wastes memory prewarming non-critical tasks with ample slack, while starving critical bottleneck tasks. Furthermore, SSC assigns critical tasks to the fastest node even if that node has no warm container, incurring a 2.2-second cold start on the critical path. AP-CWS incorporates warm container status ($y_{ij} s_i$) directly into the node selection formula.

### Q2: *"How does AP-CWS avoid wasting memory on unnecessary warm containers?"*
**Answer:**  
AP-CWS uses **Criticality-Aware Cold-Start Budgeting**:
$$\varepsilon_i = \varepsilon_{\max} - (\varepsilon_{\max} - \varepsilon_{\min}) \cdot K_i$$
Non-critical functions with large slack ($K_i \approx 0.2$) are assigned a loose cold-start tolerance ($\varepsilon_{\max} = 0.25$), requiring small or zero warm pools. Only critical-path functions ($K_i = 1.0$) receive tight tolerance ($\varepsilon_{\min} = 0.02$). This saves up to 40% idle memory compared to uniform keep-alive.

### Q3: *"Can your scheduler scale to a production cluster of hundreds of nodes?"*
**Answer:**  
Yes! That is exactly what our **Fast-AP-CWS** implementation demonstrates in Option 4. By applying the Power-of-Two-Choices theorem ($d=2$) and memoizing DAG topological depth ($K_i$), we decouple dispatch complexity from cluster size $|J|$. Dispatch overhead is only **13 microseconds per task** on 128 nodes, enabling over 77,000 scheduling decisions per second on a single thread.

---

## 5. Repository File Map for Reference

| Directory / File | Description |
| :--- | :--- |
| [`src/demo.py`](../../src/demo.py) | **Interactive CLI demo script** (Options 1–6). Execute with `python src/demo.py`. |
| [`docs/reports/DEMO_GUIDE.md`](DEMO_GUIDE.md) | **This presentation guide** with speaking scripts and Q&A answers. |
| [`paper/AP_CWS_FULL_CONFERENCE_PAPER.md`](../../paper/AP_CWS_FULL_CONFERENCE_PAPER.md) | **Full conference paper** containing mathematical proofs, literature review, and tables (PDF: [`ap_cws_paper.pdf`](../../paper/ap_cws_paper.pdf)). |
| [`docs/figures/`](../figures) | High-resolution diagrams (`diagram1`–`diagram5`) and benchmark charts (`fig1`–`fig8`) in 300 DPI PNG. |
| [`src/results/`](../../src/results) | Empirical simulation raw data and aggregated CSV summary tables. |
| [`src/ap_cws_simpy_engine.py`](../../src/ap_cws_simpy_engine.py) | SimPy discrete-event simulation engine with container lifecycle and heterogeneous node models. |
| [`src/ap_cws_baselines.py`](../../src/ap_cws_baselines.py) | Implementation of AP-CWS, SSC 2024, CASR 2025, MPC 2025, DROPS 2026, and ablations. |
| [`src/run_2nd_evaluation.py`](../../src/run_2nd_evaluation.py) | Benchmark runner for the 324-run experimental suite. |
| [`src/run_scalability_stress_test.py`](../../src/run_scalability_stress_test.py) | Scalability stress testing and Fast-AP-CWS benchmark runner. |
| [`src/ap_cws_tests.py`](../../src/ap_cws_tests.py) | Unit and integration test suite (`python src/ap_cws_tests.py`). |
