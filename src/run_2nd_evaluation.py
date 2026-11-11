"""
AP-CWS: 2nd Evaluation Comprehensive Benchmarking Suite
Executes simulation across 3 realistic workflow DAGs, 4 workload arrival patterns,
and 9 algorithmic configurations (Proposed AP-CWS, 4 latest literature baselines, 1 default, 3 ablations).
Generates empirical CSV/JSON datasets and 6 high-resolution publication figures.
"""

import os
import sys
import json
import time
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SRC_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "docs", "figures")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ap_cws_simpy_engine import (
    FunctionSpec,
    WorkflowDefinition,
    ClusterSimulator,
    SimulationNode
)
from ap_cws_baselines import (
    compute_dag_metrics,
    APCWSSchedulerPolicy,
    SSCSchedulerPolicy,
    CASRSchedulerPolicy,
    MPCSchedulerPolicy,
    DROPSSchedulerPolicy,
    FixedKeepAlivePolicy,
    APCWSAblationNoCriticality,
    APCWSAblationNoPrewarm,
    APCWSAblationNoCost
)


# ============================================================================
# 1. WORKFLOW DAG DEFINITIONS
# ============================================================================

def create_video_analytics_dag() -> WorkflowDefinition:
    """
    DAG 1: Video Analytics Pipeline (8 functions).
    Decode -> Split -> [ObjDetect (critical), FaceRec, OCR (slack), Audio] -> Merge -> Annotate.
    """
    functions = {
        "decode": FunctionSpec("decode", memory_mb=256, cpu_units=0.5, base_duration=0.4, cold_start_delay=1.2, parents=[], children=["split"]),
        "split": FunctionSpec("split", memory_mb=256, cpu_units=0.5, base_duration=0.3, cold_start_delay=0.8, parents=["decode"], children=["obj_detect", "face_rec", "plate_ocr", "audio_tr"]),
        "obj_detect": FunctionSpec("obj_detect", memory_mb=1024, cpu_units=2.0, base_duration=1.8, cold_start_delay=2.5, parents=["split"], children=["merge"]),
        "face_rec": FunctionSpec("face_rec", memory_mb=512, cpu_units=1.0, base_duration=1.2, cold_start_delay=1.8, parents=["split"], children=["merge"]),
        "plate_ocr": FunctionSpec("plate_ocr", memory_mb=256, cpu_units=0.5, base_duration=0.5, cold_start_delay=1.0, parents=["split"], children=["merge"]),
        "audio_tr": FunctionSpec("audio_tr", memory_mb=512, cpu_units=1.0, base_duration=1.0, cold_start_delay=1.5, parents=["split"], children=["merge"]),
        "merge": FunctionSpec("merge", memory_mb=512, cpu_units=1.0, base_duration=0.4, cold_start_delay=0.9, parents=["obj_detect", "face_rec", "plate_ocr", "audio_tr"], children=["annotate"]),
        "annotate": FunctionSpec("annotate", memory_mb=256, cpu_units=0.5, base_duration=0.3, cold_start_delay=0.8, parents=["merge"], children=[]),
    }
    return WorkflowDefinition(
        id="Video_Analytics_8F",
        functions=functions,
        entry_nodes=["decode"],
        exit_nodes=["annotate"],
        slo_deadline=4.5  # seconds
    )


def create_scientific_montage_dag() -> WorkflowDefinition:
    """
    DAG 2: Scientific Workflow - Astronomical Montage Mosaic (10 functions).
    Project -> [Diff1, Diff2, Diff3] -> FitBg -> [Corr1, Corr2, Corr3] -> Coadd.
    """
    functions = {
        "project": FunctionSpec("project", memory_mb=512, cpu_units=1.0, base_duration=0.8, cold_start_delay=1.5, parents=[], children=["diff_1", "diff_2", "diff_3"]),
        "diff_1": FunctionSpec("diff_1", memory_mb=512, cpu_units=1.0, base_duration=1.2, cold_start_delay=1.8, parents=["project"], children=["fit_bg"]),
        "diff_2": FunctionSpec("diff_2", memory_mb=512, cpu_units=1.0, base_duration=1.4, cold_start_delay=1.9, parents=["project"], children=["fit_bg"]),
        "diff_3": FunctionSpec("diff_3", memory_mb=512, cpu_units=1.0, base_duration=1.1, cold_start_delay=1.7, parents=["project"], children=["fit_bg"]),
        "fit_bg": FunctionSpec("fit_bg", memory_mb=1024, cpu_units=2.0, base_duration=1.5, cold_start_delay=2.2, parents=["diff_1", "diff_2", "diff_3"], children=["corr_1", "corr_2", "corr_3"]),
        "corr_1": FunctionSpec("corr_1", memory_mb=512, cpu_units=1.0, base_duration=1.0, cold_start_delay=1.6, parents=["fit_bg"], children=["coadd"]),
        "corr_2": FunctionSpec("corr_2", memory_mb=512, cpu_units=1.0, base_duration=1.0, cold_start_delay=1.6, parents=["fit_bg"], children=["coadd"]),
        "corr_3": FunctionSpec("corr_3", memory_mb=512, cpu_units=1.0, base_duration=1.2, cold_start_delay=1.8, parents=["fit_bg"], children=["coadd"]),
        "coadd": FunctionSpec("coadd", memory_mb=1024, cpu_units=2.0, base_duration=1.6, cold_start_delay=2.4, parents=["corr_1", "corr_2", "corr_3"], children=[]),
    }
    return WorkflowDefinition(
        id="Montage_Scientific_9F",
        functions=functions,
        entry_nodes=["project"],
        exit_nodes=["coadd"],
        slo_deadline=7.0  # seconds
    )


def create_ecommerce_dag() -> WorkflowDefinition:
    """
    DAG 3: E-Commerce Checkout Flow (6 functions).
    OrderValidate -> [InventoryCheck, FraudDetection (critical), CreditCheck] -> Payment -> Notify.
    """
    functions = {
        "validate": FunctionSpec("validate", memory_mb=128, cpu_units=0.5, base_duration=0.2, cold_start_delay=0.6, parents=[], children=["inventory", "fraud", "credit"]),
        "inventory": FunctionSpec("inventory", memory_mb=256, cpu_units=0.5, base_duration=0.4, cold_start_delay=0.9, parents=["validate"], children=["payment"]),
        "fraud": FunctionSpec("fraud", memory_mb=1024, cpu_units=2.0, base_duration=1.4, cold_start_delay=2.2, parents=["validate"], children=["payment"]),
        "credit": FunctionSpec("credit", memory_mb=256, cpu_units=0.5, base_duration=0.5, cold_start_delay=1.0, parents=["validate"], children=["payment"]),
        "payment": FunctionSpec("payment", memory_mb=512, cpu_units=1.0, base_duration=0.8, cold_start_delay=1.4, parents=["inventory", "fraud", "credit"], children=["notify"]),
        "notify": FunctionSpec("notify", memory_mb=128, cpu_units=0.5, base_duration=0.2, cold_start_delay=0.5, parents=["payment"], children=[]),
    }
    return WorkflowDefinition(
        id="ECommerce_6F",
        functions=functions,
        entry_nodes=["validate"],
        exit_nodes=["notify"],
        slo_deadline=3.4  # seconds
    )


# ============================================================================
# 2. WORKLOAD TRACE GENERATORS (Azure Functions Patterns)
# ============================================================================

def generate_arrival_trace(pattern: str, total_time: float = 300.0, seed: int = 42) -> List[float]:
    """
    Generates workflow arrival timestamps (seconds) under various real-world traffic profiles.
    """
    rng = np.random.default_rng(seed)
    arrivals = []
    current_time = 0.0

    if pattern == "stable":
        # Poisson process with steady rate lambda = 1.0 req/s
        rate = 1.0
        while current_time < total_time:
            inter = rng.exponential(1.0 / rate)
            current_time += inter
            if current_time < total_time:
                arrivals.append(current_time)

    elif pattern == "rising":
        # Linearly increasing arrival rate (0.4 req/s -> 2.4 req/s)
        while current_time < total_time:
            progress = current_time / total_time
            rate = 0.4 + 2.0 * progress
            inter = rng.exponential(1.0 / rate)
            current_time += inter
            if current_time < total_time:
                arrivals.append(current_time)

    elif pattern == "falling":
        # Linearly decreasing arrival rate (2.4 req/s -> 0.4 req/s)
        while current_time < total_time:
            progress = current_time / total_time
            rate = 2.4 - 2.0 * progress
            inter = rng.exponential(1.0 / rate)
            current_time += inter
            if current_time < total_time:
                arrivals.append(current_time)

    elif pattern == "bursty":
        # Spiky / bursty arrival (Pareto-distributed burst sizes + background rate)
        base_rate = 0.6
        burst_prob = 0.08  # burst trigger probability
        while current_time < total_time:
            if rng.random() < burst_prob:
                burst_size = int(rng.pareto(a=2.0) + 4)
                for _ in range(min(burst_size, 12)):
                    current_time += rng.uniform(0.02, 0.15)
                    if current_time < total_time:
                        arrivals.append(current_time)
            else:
                inter = rng.exponential(1.0 / base_rate)
                current_time += inter
                if current_time < total_time:
                    arrivals.append(current_time)

    return arrivals


# ============================================================================
# 3. EXPERIMENTAL HARNESS
# ============================================================================

def get_cluster_nodes() -> List[Dict[str, Any]]:
    """Heterogeneous 4-node cluster with 48 GB total RAM."""
    return [
        {"id": "node_1_small", "memory_mb": 8192, "speed": 1.0, "price_per_hour": 0.05},
        {"id": "node_2_medium", "memory_mb": 8192, "speed": 1.25, "price_per_hour": 0.065},
        {"id": "node_3_fast", "memory_mb": 16384, "speed": 1.5, "price_per_hour": 0.10},
        {"id": "node_4_ultra", "memory_mb": 16384, "speed": 1.8, "price_per_hour": 0.125},
    ]


def run_benchmark_experiments():
    """Runs all benchmark evaluations and outputs metrics and charts."""
    print("=" * 80)
    print("AP-CWS 2nd Evaluation: Running Full Empirical Simulation Suite")
    print("=" * 80)

    workflows = {
        "Video Analytics (8F)": create_video_analytics_dag(),
        "Montage Scientific (9F)": create_scientific_montage_dag(),
        "E-Commerce (6F)": create_ecommerce_dag(),
    }

    patterns = ["stable", "rising", "falling", "bursty"]
    seeds = [42, 101, 2024]
    duration = 300.0  # 5 minutes per run

    def instantiate_policies():
        return [
            APCWSSchedulerPolicy(),
            SSCSchedulerPolicy(),
            CASRSchedulerPolicy(),
            MPCSchedulerPolicy(),
            DROPSSchedulerPolicy(),
            FixedKeepAlivePolicy(),
            APCWSAblationNoCriticality(),
            APCWSAblationNoPrewarm(),
            APCWSAblationNoCost()
        ]

    results_data = []

    total_runs = len(workflows) * len(patterns) * len(instantiate_policies()) * len(seeds)
    run_idx = 0

    print(f"Total benchmark executions scheduled: {total_runs}")
    start_wall = time.time()

    for wf_name, wf_def in workflows.items():
        _, _, _, critical_path = compute_dag_metrics(wf_def)
        print(f"\n--> Workflow: {wf_name} | Critical Path: {critical_path} | Target SLO: {wf_def.slo_deadline}s")

        for pattern in patterns:
            for seed in seeds:
                arrivals = generate_arrival_trace(pattern, total_time=duration, seed=seed)
                policies = instantiate_policies()

                for policy in policies:
                    run_idx += 1
                    # Initialize simulator
                    cluster_nodes = get_cluster_nodes()
                    sim = ClusterSimulator(cluster_nodes, random_seed=seed)
                    sim.run_workload(wf_def, policy, arrivals, critical_path)
                    sim.env.run(until=duration + 20.0)

                    stats = sim.get_summary_statistics()

                    row = {
                        "workflow": wf_name,
                        "pattern": pattern,
                        "seed": seed,
                        "policy": policy.name,
                        "total_workflows": stats.get("total_workflows", 0),
                        "mean_latency": stats.get("mean_latency", 0.0),
                        "median_latency": stats.get("median_latency", 0.0),
                        "p95_latency": stats.get("p95_latency", 0.0),
                        "p99_latency": stats.get("p99_latency", 0.0),
                        "cold_start_rate_pct": stats.get("cold_start_rate_pct", 0.0),
                        "critical_path_cold_start_pct": stats.get("critical_path_cold_start_pct", 0.0),
                        "slo_violation_rate_pct": stats.get("slo_violation_rate_pct", 0.0),
                        "execution_cost_usd": stats.get("execution_cost_usd", 0.0),
                        "idle_gb_hours": stats.get("idle_gb_hours", 0.0),
                        "total_cost_usd": stats.get("total_cost_usd", 0.0),
                    }
                    results_data.append(row)

                    if run_idx % 30 == 0 or run_idx == total_runs:
                        pct = (run_idx / total_runs) * 100.0
                        elapsed = time.time() - start_wall
                        print(f"[{pct:5.1f}% complete] Run {run_idx}/{total_runs} - {policy.name:25s} | P95: {row['p95_latency']:.2f}s | Cold: {row['cold_start_rate_pct']:.1f}% | Elapsed: {elapsed:.1f}s")

    df = pd.DataFrame(results_data)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_csv = os.path.join(RESULTS_DIR, "simulation_results_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\nSaved raw simulation results to '{raw_csv}'")

    # ========================================================================
    # 4. AGGREGATE SUMMARY TABLES
    # ========================================================================
    metrics_to_agg = [
        "mean_latency", "median_latency", "p95_latency",
        "cold_start_rate_pct", "critical_path_cold_start_pct",
        "slo_violation_rate_pct", "idle_gb_hours", "total_cost_usd"
    ]

    summary_df = df.groupby(["workflow", "policy"])[metrics_to_agg].agg(["mean", "std"]).reset_index()
    summary_csv = os.path.join(RESULTS_DIR, "simulation_results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary results to '{summary_csv}'")

    overall_df = df.groupby("policy")[metrics_to_agg].agg(["mean", "std"]).reset_index()
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY METRICS (Mean across all Workflows, Patterns, and Seeds)")
    print("=" * 90)
    for p in overall_df["policy"].unique():
        p_row = overall_df[overall_df["policy"] == p]
        p95_m = p_row[("p95_latency", "mean")].values[0]
        p95_s = p_row[("p95_latency", "std")].values[0]
        cold_m = p_row[("cold_start_rate_pct", "mean")].values[0]
        crit_cold_m = p_row[("critical_path_cold_start_pct", "mean")].values[0]
        slo_m = p_row[("slo_violation_rate_pct", "mean")].values[0]
        cost_m = p_row[("total_cost_usd", "mean")].values[0]
        idle_m = p_row[("idle_gb_hours", "mean")].values[0]

        print(f"{p:30s} | P95: {p95_m:5.2f}±{p95_s:4.2f}s | Cold: {cold_m:5.1f}% | CritCold: {crit_cold_m:5.1f}% | SLO Viol: {slo_m:5.1f}% | IdleGBh: {idle_m:5.3f} | Cost: ${cost_m:7.5f}")

    # ========================================================================
    # 5. GENERATE PUBLICATION CHARTS
    # ========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    core_policies = [
        "AP-CWS (Proposed)",
        "SSC (Pan 2024)",
        "CASR (Chen 2025)",
        "MPC (Nguyen 2025)",
        "DROPS (EuroSys 2026)",
        "Fixed 10-min Keep-Alive"
    ]
    os.makedirs(FIGURES_DIR, exist_ok=True)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77", "#666666"]

    # --- FIGURE 1: P95 Latency Comparison ---
    fig, ax = plt.subplots(figsize=(11, 6))
    wf_names = list(workflows.keys())
    x = np.arange(len(wf_names))
    width = 0.13

    for i, pol in enumerate(core_policies):
        means = []
        errs = []
        for wf in wf_names:
            sub = df[(df["policy"] == pol) & (df["workflow"] == wf)]
            means.append(sub["p95_latency"].mean())
            errs.append(sub["p95_latency"].std())
        ax.bar(x + (i - 2.5) * width, means, width, yerr=errs, capsize=3, label=pol, color=colors[i], alpha=0.9)

    ax.set_ylabel("P95 Workflow End-to-End Latency (seconds)", fontweight="bold")
    ax.set_title("Figure 1: P95 Latency Across Serverless Workflow Topologies (Lower is Better)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(wf_names, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_p95_latency_comparison.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig1_p95_latency_comparison.png')}'")

    # --- FIGURE 2: Cold Start Rate (Overall vs Critical Path) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    pol_means_all = [df[df["policy"] == pol]["cold_start_rate_pct"].mean() for pol in core_policies]
    pol_means_crit = [df[df["policy"] == pol]["critical_path_cold_start_pct"].mean() for pol in core_policies]
    pol_short_names = ["AP-CWS\n(Proposed)", "SSC\n(Pan 2024)", "CASR\n(Chen 2025)", "MPC\n(Nguyen 2025)", "DROPS\n(EuroSys 2026)", "Fixed\nKeep-Alive"]

    bars1 = ax1.bar(pol_short_names, pol_means_all, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Overall Cold Start Rate (%)", fontweight="bold")
    ax1.set_title("Overall Cold Start Rate Across All Functions", fontweight="bold")
    ax1.set_ylim(0, max(pol_means_all) * 1.25)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

    bars2 = ax2.bar(pol_short_names, pol_means_crit, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Critical-Path Cold Start Rate (%)", fontweight="bold")
    ax2.set_title("Critical Path Cold Start Rate (Direct Latency Penalty)", fontweight="bold")
    ax2.set_ylim(0, max(pol_means_crit) * 1.25)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.suptitle("Figure 2: Cold Start Frequency (Overall vs Critical Path Isolation)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_cold_start_rates.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig2_cold_start_rates.png')}'")

    # --- FIGURE 3: Cost vs Idle Memory Wastage ---
    fig, ax = plt.subplots(figsize=(10, 6))
    idle_hours = [df[df["policy"] == pol]["idle_gb_hours"].mean() for pol in core_policies]
    total_costs = [df[df["policy"] == pol]["total_cost_usd"].mean() * 1000 for pol in core_policies]  # in m$ (milli-dollars)

    for i, pol in enumerate(core_policies):
        ax.scatter(idle_hours[i], total_costs[i], color=colors[i], s=260, zorder=4, edgecolor="black", linewidth=1.5, label=pol)
        # Annotation offset
        ax.annotate(pol_short_names[i].replace("\n", " "), (idle_hours[i], total_costs[i]), textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight="bold")

    ax.set_xlabel("Idle Warm Memory Wastage (GB-Hours)", fontweight="bold")
    ax.set_ylabel("Total Operational Cost (milli-$ per 5-min workload)", fontweight="bold")
    ax.set_title("Figure 3: Pareto Trade-off: Operational Cost vs Memory Footprint", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_cost_vs_idle_memory.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig3_cost_vs_idle_memory.png')}'")

    # --- FIGURE 4: SLO Violation Rate Across Traffic Patterns ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pat_labels = ["Stable", "Rising (Rush)", "Falling", "Bursty (Spiky)"]
    x_pat = np.arange(len(pat_labels))
    width = 0.13

    for i, pol in enumerate(core_policies):
        pat_means = []
        for pat in ["stable", "rising", "falling", "bursty"]:
            val = df[(df["policy"] == pol) & (df["pattern"] == pat)]["slo_violation_rate_pct"].mean()
            pat_means.append(val)
        ax.bar(x_pat + (i - 2.5) * width, pat_means, width, label=pol, color=colors[i], alpha=0.9)

    ax.set_ylabel("SLO Violation Rate (%)", fontweight="bold")
    ax.set_title("Figure 4: SLO Violation Robustness Across Real-world Arrival Patterns", fontweight="bold", pad=12)
    ax.set_xticks(x_pat)
    ax.set_xticklabels(pat_labels, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_slo_violation_rate.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig4_slo_violation_rate.png')}'")

    # --- FIGURE 5: Ablation Analysis of AP-CWS Components ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ablation_pols = [
        "AP-CWS (Proposed)",
        "AP-CWS (No Criticality)",
        "AP-CWS (No JIT Prewarm)",
        "AP-CWS (No Cost Weight)"
    ]
    abl_colors = ["#2b5c8f", "#e6ab02", "#e7298a", "#66a61e"]
    abl_labels = ["Full AP-CWS", "w/o Criticality (K=0)", "w/o JIT Prewarm", "w/o Cost Weight (μ=0)"]

    abl_p95 = [df[df["policy"] == pol]["p95_latency"].mean() for pol in ablation_pols]
    abl_cost = [df[df["policy"] == pol]["total_cost_usd"].mean() * 1000 for pol in ablation_pols]

    bars_abl1 = ax1.bar(abl_labels, abl_p95, color=abl_colors, alpha=0.88, edgecolor="black")
    ax1.set_ylabel("P95 Latency (s)", fontweight="bold")
    ax1.set_title("Ablation: P95 Latency Degradation", fontweight="bold")
    for b in bars_abl1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 0.08, f"{yval:.2f}s", ha='center', va='bottom', fontweight="bold")

    bars2_abl = ax2.bar(abl_labels, abl_cost, color=abl_colors, alpha=0.88, edgecolor="black")
    ax2.set_ylabel("Total Operational Cost (milli-$)", fontweight="bold")
    ax2.set_title("Ablation: Cost Impact", fontweight="bold")
    for b in bars2_abl:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 0.005, f"${yval:.3f}", ha='center', va='bottom', fontweight="bold")

    plt.suptitle("Figure 5: Component Ablation Study of AP-CWS Mechanisms", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_ablation_study.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig5_ablation_study.png')}'")

    # --- FIGURE 6: Critical Path vs Non-Critical Path Cold Start Decomposition ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(len(core_policies))
    w_bar = 0.35

    ax.bar(x_indices - w_bar/2, pol_means_crit, w_bar, label="Critical Path Cold Start %", color="#d95f02", alpha=0.9, edgecolor="black")
    ax.bar(x_indices + w_bar/2, pol_means_all, w_bar, label="Overall Cold Start %", color="#7570b3", alpha=0.9, edgecolor="black")

    ax.set_ylabel("Cold Start Rate (%)", fontweight="bold")
    ax.set_title("Figure 6: Critical Path Cold Starts vs Overall Cold Starts", fontweight="bold", pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(pol_short_names, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_critical_path_cold_starts.png"), dpi=300)
    plt.close()
    print(f"Generated '{os.path.join(FIGURES_DIR, 'fig6_critical_path_cold_starts.png')}'")

    print("\n" + "=" * 80)
    print("AP-CWS 2nd Evaluation Benchmark Completed Successfully!")
    print(f"Total Wall Clock Time: {time.time() - start_wall:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_experiments()
