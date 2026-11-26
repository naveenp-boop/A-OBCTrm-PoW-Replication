"""Generate publication-grade Figures 1, 2, 3, 4, and 5 directly from simulated results & architecture spec."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
PAPER_ROOT = ROOT.parent  # Also sync directly to workspace manuscript folder


def generate_architecture_diagram() -> None:
    """Render Figure 1 cleanly with exact paper notation (normalized drivers, c_k, probation, T(t), feedback loops)."""
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=300)
    ax.set_xlim(-1.2, 11.2)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    def box(cx, cy, w, h, title, sub="", fc="#F8FAFC", ec="#1E3A8A", tc="#0F172A", title_fs=9.8, sub_fs=7.6, bold_fc=False):
        patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
                               facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(patch)
        if sub:
            ax.text(cx, cy + h * 0.19, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=tc)
            ax.text(cx, cy - h * 0.21, sub, ha="center", va="center", fontsize=sub_fs, color="#1E293B" if not bold_fc else tc)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=tc)

    def arrow(x0, y0, x1, y1, color="#334155", rad=0.0):
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_width=0.22,head_length=0.32",
                              connectionstyle=f"arc3,rad={rad}", linewidth=1.6, color=color)
        ax.add_patch(arr)

    box(5.0, 10.55, 8.8, 0.72, "A-OBCTrm-PoW: Trust-Aware Consensus Admission Layer",
        fc="#1D4ED8", ec="#1E3A8A", tc="white", title_fs=11.2, bold_fc=True)

    box(5.0, 9.20, 6.2, 0.92, "Miner Inputs & Neighborhood Graph $\\mathcal{N}(i)$",
        "Transaction Streams  |  Peer Ratings $r_{k\\to i}$  |  Probation ($\\gamma=0.50,\\; N_{\\min}=50$)",
        fc="#F1F5F9", ec="#475569")
    arrow(5.0, 10.16, 5.0, 9.70)

    box(5.0, 7.75, 6.2, 0.98, "Trust Metric Evaluation Engine",
        "$\\mathrm{THS}_i$ (Historical)  |  $\\mathrm{NSI}_i$ (Stability)  |  $\\mathrm{PRF}_i$ with $c_k = \\min(1, \\mathrm{THS}_k / \\theta_c)$",
        fc="#FEF3C7", ec="#B45309")
    arrow(5.0, 8.71, 5.0, 8.27)

    box(5.0, 6.25, 6.2, 1.02, "Adaptive Weight Computation ($W = 50$ Sliding Window)",
        "Drivers $(\\hat{\\lambda}, \\hat{\\tau}, \\hat{\\delta}) \\in [0,1]^3 \\to \\exp(k_1\\hat{\\delta}),\\; 1+k_2\\hat{\\lambda},\\; 1+k_3\\hat{\\tau}$  ($w_{\\max}=0.80$)",
        fc="#E0F2FE", ec="#0369A1")
    arrow(5.0, 7.23, 5.0, 6.79)

    box(5.0, 4.78, 6.2, 0.94, "TOPSIS Closeness Ranking $C_i \\in [0, 1]$",
        "Euclidean Separation $S_i^+, S_i^-$  |  Probationary Cap $w_{\\mathrm{THS}} \\leq 0.30$ (first 50 blocks)",
        fc="#DCFCE7", ec="#15803D")
    arrow(5.0, 5.71, 5.0, 5.28)

    box(5.0, 3.30, 6.2, 0.96, "Dynamic 5%-Trimmed Threshold & Pool Partition",
        "Quickselect $T(t) = \\mathrm{clip}(\\mu_{\\mathrm{trim}} - \\alpha\\sigma_{\\mathrm{trim}} + \\beta\\hat{\\delta},\\; 0.47,\\; 0.56) \\to \\mathcal{M}_{\\mathrm{HM}}$",
        fc="#FFEDD5", ec="#C2410C")
    arrow(5.0, 4.28, 5.0, 3.81)

    box(5.0, 1.88, 6.2, 0.88, "Filtered Proof-of-Work & Block Validation ($\\mathcal{M}_{\\mathrm{HM}}$)",
        "Lower Mempool Congestion  |  Fewer Forks  |  Faster Compact-Block Relay",
        fc="#FEE2E2", ec="#B91C1C")
    arrow(5.0, 2.79, 5.0, 2.35)

    box(5.0, 0.55, 6.2, 0.84, "Ledger Append & Outcome Recording $\\phi_i(t) \\in \\{0, 1\\}$",
        "Valid: $+\\eta(1-\\mathrm{THS}_i)$ ($\\eta=0.05$)   |   Fault: $\\max(0,\\; \\kappa\\,\\mathrm{THS}_i - \\zeta)$ ($\\kappa=0.40$)",
        fc="#E2E8F0", ec="#334155")
    arrow(5.0, 1.41, 5.0, 1.00)

    # Feedback loop 1 (left): network telemetry back to adaptive weights (kept completely outside boxes)
    arrow(1.86, 1.88, 1.86, 6.25, color="#0369A1", rad=-0.26)
    ax.text(0.05, 4.05, "Sliding-Window ($W=50$)\nTelemetry Feedback\n$(\\hat{\\lambda},\\; \\hat{\\tau},\\; \\hat{\\delta})$",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0369A1")

    # Feedback loop 2 (right): asymmetric THS update back to Trust Metric Evaluation Engine
    arrow(8.14, 0.55, 8.14, 7.75, color="#B45309", rad=0.25)
    ax.text(9.95, 4.15, "Asymmetric Trust\nFeedback Loop\n$\\mathrm{THS}_i(t+1)$\n($>60\\%$ collapse)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#B45309")

    plt.tight_layout()
    for dest in (FIG_DIR / "fig1_architecture.png", PAPER_ROOT / "extracted_fig_p5_0.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_architecture_diagram()
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
    for dest in (FIG_DIR / "fig3_tps_comparison.png", PAPER_ROOT / "tps_comparison.png"):
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
    for dest in (FIG_DIR / "fig4_trust_distribution.png", PAPER_ROOT / "trust_distribution.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 5: Measured Computational Overhead / CPU Utilization (%) — all 7 models
    # =========================================================================
    cpu_models = [
        ("Standard PoW", "Standard PoW (No Defense)"),
        ("Token Bucket", "Rate Limiting (Token Bucket)"),
        ("Dynamic Fee", "Dynamic Fee Mechanism"),
        ("EWMA Filter", "EWMA Reputation Filter"),
        ("Static TOPSIS", "Static TOPSIS (Pre-processing)"),
        ("A-OBCTrm-PoW", "A-OBCTrm-PoW (Proposed)"),
        ("ML-IDS", "ML-Based IDS (Network Layer)"),
    ]
    c_labels = [m[0] for m in cpu_models]
    c_means = [data["baseline_ddos_comparison"][m[1]]["cpu_overhead_pct"]["mean"] for m in cpu_models]
    c_sds = [data["baseline_ddos_comparison"][m[1]]["cpu_overhead_pct"]["sd"] for m in cpu_models]
    c_colors = ["#718096", "#A0AEC0", "#CBD5E0", "#4A5568", "#38A169", "#2B6CB0", "#E53E3E"]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=300)
    x = np.arange(len(c_labels))
    bars = ax.bar(x, c_means, yerr=c_sds, capsize=4, width=0.58, color=c_colors, edgecolor="#1A202C", alpha=0.90)
    for bar, m, sd in zip(bars, c_means, c_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 1.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=16, ha="right")
    ax.set_ylabel("Average CPU Utilization (%)")
    ax.set_title("Measured Single-Core Validation Slot CPU Utilization (All 7 Models, n=20)")
    ax.set_ylim(0, max(36, max(c_means) + max(c_sds) + 8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for dest in (FIG_DIR / "fig5_cpu_utilization.png", PAPER_ROOT / "cpu_utilization.png"):
        plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated Figures 1, 2, 3, 4, and 5 in {FIG_DIR} and synced to {PAPER_ROOT}")


if __name__ == "__main__":
    main()
