"""Generate publication-grade Figures 2, 3, 4, and 5 directly from simulated results JSON."""

from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
PAPER_ROOT = ROOT.parent  # Also sync directly to workspace manuscript folder


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "all_experiment_results.json", encoding="utf-8") as f:
        data = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
    })

    # =========================================================================
    # FIGURE 2: Reconciliation of Block Creation Time (Nominal vs Active DDoS)
    # =========================================================================
    models_fig2 = ["A-OBCTrm-PoW", "Static TOPSIS", "ML-IDS", "Standard PoW"]
    label_map = {
        "A-OBCTrm-PoW": "A-OBCTrm-PoW (Proposed)",
        "Static TOPSIS": "Static TOPSIS (Pre-processing)",
        "ML-IDS": "ML-Based IDS (Network Layer)",
        "Standard PoW": "Standard PoW (No Defense)",
    }
    nom_means = [data["nominal_summary"][m]["b_cr_s"]["mean"] for m in models_fig2]
    nom_sds = [data["nominal_summary"][m]["b_cr_s"]["sd"] for m in models_fig2]
    ddos_means = [data["baseline_ddos_comparison"][label_map[m]]["b_cr_s"]["mean"] for m in models_fig2]
    ddos_sds = [data["baseline_ddos_comparison"][label_map[m]]["b_cr_s"]["sd"] for m in models_fig2]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    x = np.arange(len(models_fig2))
    w = 0.36
    b1 = ax.bar(x - w / 2, nom_means, w, yerr=nom_sds, capsize=4, label="Nominal / Quiescent Regime", color="#2B6CB0", edgecolor="#1A365D", alpha=0.92)
    b2 = ax.bar(x + w / 2, ddos_means, w, yerr=ddos_sds, capsize=4, label="Active DDoS Regime (5,000 tx/s)", color="#E53E3E", edgecolor="#742A2A", alpha=0.88)

    for bar, m, sd in zip(b1, nom_means, nom_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.25, f"{m:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1A365D")
    for bar, m, sd in zip(b2, ddos_means, ddos_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.25, f"{m:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#742A2A")

    ax.set_xticks(x)
    ax.set_xticklabels(models_fig2)
    ax.set_ylabel("Block Creation Decision Latency $B_{cr}$ (s)")
    ax.set_title("Block Creation Time ($B_{cr}$) Across Nominal vs. Active DDoS Regimes (30 Trials)")
    ax.set_ylim(0, max(ddos_means) + max(ddos_sds) + 2.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(framealpha=0.95, loc="upper left")
    plt.tight_layout()
    for dest in (FIG_DIR / "fig2_bcr_reconciliation.png", PAPER_ROOT / "extracted_fig_p10_0.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: System Throughput (TPS) Resilience Over Time (mean ± 1 SD)
    # =========================================================================
    t_axis = np.arange(len(data["tps_curves"]["A-OBCTrm-PoW"]["mean"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=300)
    styles = [
        ("A-OBCTrm-PoW", "#1F77B4", "o", "-"),
        ("ML-IDS", "#FF7F0E", "s", "-"),
        ("Static TOPSIS", "#2CA02C", "^", "--"),
        ("Standard PoW", "#D62728", "v", "-"),
    ]
    for name, col, mk, ls in styles:
        m_arr = np.asarray(data["tps_curves"][name]["mean"])
        s_arr = np.asarray(data["tps_curves"][name]["sd"])
        ax.plot(t_axis, m_arr, color=col, linestyle=ls, linewidth=2.0, label=name)
        ax.fill_between(t_axis, np.maximum(0, m_arr - s_arr), m_arr + s_arr, color=col, alpha=0.14)
        # Mark every 20s cleanly
        pts = np.arange(0, len(t_axis), 20)
        ax.plot(t_axis[pts], m_arr[pts], color=col, marker=mk, linestyle="None", markersize=5.5)

    ax.axvline(30, color="black", linestyle="--", linewidth=1.6, label="DDoS Initiated (t = 30 s)")
    ax.set_xlabel("Simulation Time (seconds)")
    ax.set_ylabel("Honest Validated Goodput (TPS)")
    ax.set_title("System Throughput Resilience Under 5,000 tx/s DDoS Flooding (Mean ± 1 SD, n=30)")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 3250)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.56), framealpha=0.95)
    plt.tight_layout()
    for dest in (FIG_DIR / "fig3_tps_comparison.png", PAPER_ROOT / "tps_comparison.png", PAPER_ROOT / "extracted_fig_p11_0.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: Distribution of TOPSIS Closeness Scores C_i (with T(t) & KS test)
    # =========================================================================
    td = data["trust_distribution_sample"]
    hon_scores = np.asarray(td["honest_scores"])
    mal_scores = np.asarray(td["malicious_scores"])
    T_val = td["threshold_T"]
    ks_d = td["ks_D"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    bins = np.linspace(0.0, 1.0, 34)
    ax.hist(mal_scores, bins=bins, color="#D62728", alpha=0.80, edgecolor="black", linewidth=0.6, label=f"Malicious Miners (n={len(mal_scores)})")
    ax.hist(hon_scores, bins=bins, color="#2B6CB0", alpha=0.80, edgecolor="black", linewidth=0.6, label=f"Honest Miners (n={len(hon_scores)})")
    ax.axvline(T_val, color="black", linestyle="--", linewidth=2.0, label=f"Dynamic Threshold T(t) = {T_val:.3f}")
    y_top = ax.get_ylim()[1]
    ax.set_ylim(0, y_top * 1.12)
    ax.text(0.03, y_top * 0.62, f"Two-sample KS test:\nD = {ks_d:.3f}, p < 0.001", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.95), fontsize=9)
    ax.set_xlabel("TOPSIS Closeness Score ($C_i$)")
    ax.set_ylabel("Number of Miners")
    ax.set_title("Miner Trust Score ($C_i$) Distribution Under 30% Adversarial Load")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    for dest in (FIG_DIR / "fig4_trust_distribution.png", PAPER_ROOT / "trust_distribution.png", PAPER_ROOT / "extracted_fig_p12_0.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 5: Measured Computational Overhead / CPU Utilization (%)
    # =========================================================================
    cpu_models = [
        ("Standard PoW", "Standard PoW (No Defense)"),
        ("EWMA Filter", "EWMA Reputation Filter"),
        ("A-OBCTrm-PoW", "A-OBCTrm-PoW (Proposed)"),
        ("Static TOPSIS", "Static TOPSIS (Pre-processing)"),
        ("ML-IDS", "ML-Based IDS (Network Layer)"),
    ]
    c_labels = [m[0] for m in cpu_models]
    c_means = [data["baseline_ddos_comparison"][m[1]]["cpu_overhead_pct"]["mean"] for m in cpu_models]
    c_sds = [data["baseline_ddos_comparison"][m[1]]["cpu_overhead_pct"]["sd"] for m in cpu_models]
    c_colors = ["#718096", "#4A5568", "#2B6CB0", "#38A169", "#E53E3E"]

    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=300)
    x = np.arange(len(c_labels))
    bars = ax.bar(x, c_means, yerr=c_sds, capsize=4, width=0.56, color=c_colors, edgecolor="#1A202C", alpha=0.90)
    for bar, m, sd in zip(bars, c_means, c_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 1.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=12, ha="right")
    ax.set_ylabel("Average CPU Utilization (%)")
    ax.set_title("Measured Single-Core Validation Slot CPU Utilization (Mean ± SD, n=20)")
    ax.set_ylim(0, max(35, max(c_means) + max(c_sds) + 8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for dest in (FIG_DIR / "fig5_cpu_utilization.png", PAPER_ROOT / "cpu_utilization.png", PAPER_ROOT / "extracted_fig_p13_0.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Copy Figure 1 architecture into results/figures as well for completeness
    if (PAPER_ROOT / "extracted_fig_p5_0.png").exists():
        shutil.copy2(PAPER_ROOT / "extracted_fig_p5_0.png", FIG_DIR / "fig1_architecture.png")

    print(f"Generated Figures 2, 3, 4, and 5 in {FIG_DIR} and synced to {PAPER_ROOT}")


if __name__ == "__main__":
    main()
