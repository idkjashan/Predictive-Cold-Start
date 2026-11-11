# AP-CWS: Documentation, Figures & Presentation Assets

**Institution:** Indian Institute of Technology (BHU) Varanasi  
**Authors:** Nikhil Sharma, Raghav Arora, Jashanpreet Singh  
**Base Paper:** Pan et al., *"Sustainable Serverless Computing With Cold-Start Optimization and Automatic Workflow Resource Scheduling"*, IEEE Transactions on Sustainable Computing (T-SUSC), 2024.

---

## 1. Directory Overview

This `docs/` directory holds the Phase 1 presentation, demonstration guide, system diagrams, and evaluation figures generated from the simulation suite:

```text
docs/
├── README.md                              # Documentation & figures index
├── AP-CWS 1st Evaluation (5 slides).pdf   # Phase 1 milestone presentation (7 pages)
├── reports/
│   └── DEMO_GUIDE.md                      # CLI demonstration walkthrough & speaking scripts
└── figures/                               # Architectural diagrams (1–5) & benchmark plots (fig1–8)
    ├── diagram1_topological_slack.png     # Topological slack & critical-path differentiation
    ├── diagram2_workflow_topologies.png   # Evaluated DAGs (Video Analytics, Montage, E-Commerce)
    ├── diagram3_architecture.png          # Two-phase planning & event-driven dispatch loops
    ├── diagram4_operating_regimes.png     # Standard AP-CWS vs Fast-AP-CWS (O(1)) regimes
    ├── diagram5_cascading_cliff.png       # 12 req/s Cascading Latency Cliff mechanism
    ├── fig1_p95_latency_comparison.png    # P95 tail latency comparison across workflows
    ├── fig2_cold_start_rates.png          # Overall vs critical-path cold-start rate panels
    ├── fig3_cost_vs_idle_memory.png       # Monetary cost vs idle warm memory trade-off
    ├── fig4_slo_violation_rate.png        # SLO violation rates across Azure trace patterns
    ├── fig5_ablation_study.png            # Algorithmic component ablation impact
    ├── fig6_critical_path_cold_starts.png # Grouped critical-path vs overall cold-start plot
    ├── fig7_scalability_breaking_point.png# Arrival rate sweep (1–32 req/s) saturation curve
    └── fig8_scheduler_overhead_scaling.png# Dispatch latency scaling (16–256 nodes)
```

---

## 2. Related Project Assets

* **Research Paper & LaTeX Source:** See [`../paper/AP_CWS_FULL_CONFERENCE_PAPER.md`](../paper/AP_CWS_FULL_CONFERENCE_PAPER.md), [`../paper/ap_cws_paper.tex`](../paper/ap_cws_paper.tex), and the compiled [`../paper/ap_cws_paper.pdf`](../paper/ap_cws_paper.pdf).
* **Simulation Source & Benchmarks:** See [`../src/README.md`](../src/README.md) and [`../README.md`](../README.md).

---

## 3. Running Code & Regenerating Figures

From the repository root:

```bash
pip install -r requirements.txt
python src/demo.py
```

Running [`../src/run_2nd_evaluation.py`](../src/run_2nd_evaluation.py) and [`../src/run_scalability_stress_test.py`](../src/run_scalability_stress_test.py) automatically updates the CSVs in `../src/results/` and charts `fig1`–`fig8` in `docs/figures/`.
