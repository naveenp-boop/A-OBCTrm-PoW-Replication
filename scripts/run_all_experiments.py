"""Master experimental runner executing 30 independent Monte Carlo trials across all regimes,
computing rigorous paired t-tests, 95% confidence intervals, Cohen's d, KS statistics, CPU
benchmarks, and saving full JSON + CSV telemetry to results/.
"""

from __future__ import annotations
import csv
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import sklearn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import SimConfig
from sim.consensus_sim import simulate_single_trial, solve_scaled_pow_sample
from sim.baselines import MLIDSDetector
from sim.cpu_benchmark import benchmark_model_cpu_and_runtime
from scripts.verify_proofs import main as run_proofs


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def agg_metric(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def paired_stats(proposed_vals: list[float], baseline_vals: list[float]) -> dict:
    a = np.asarray(proposed_vals, dtype=np.float64)
    b = np.asarray(baseline_vals, dtype=np.float64)
    diff = b - a  # positive means proposed is faster/lower
    t_stat, p_val = stats.ttest_rel(b, a)
    n = len(diff)
    mean_diff = float(diff.mean())
    se_diff = float(diff.std(ddof=1) / np.sqrt(n))
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se_diff)
    pooled_sd = float(np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0))
    cohens_d = mean_diff / max(1e-9, pooled_sd)
    return {
        "n_trials": n,
        "mean_reduction": round(mean_diff, 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_val),
        "cohens_d": round(float(cohens_d), 4),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    cfg = SimConfig()
    ml = MLIDSDetector(seed=42)
    seeds = list(range(100, 100 + cfg.num_trials))

    print("[1/6] Running initial mathematical proof checks...")
    run_proofs()

    print("[2/6] Mining 5 sample 16-bit SHA-256 PoW blocks for ledger proof...")
    pow_blocks = []
    for blk_h in range(5):
        nonce, digest_hex, ms = solve_scaled_pow_sample(f"AOBCTrm-MainnetScaled-Block-{blk_h}", cfg.pow_leading_zero_bits)
        pow_blocks.append({"height": blk_h, "nonce": nonce, "sha256_hash": digest_hex, "solve_ms": round(ms, 2)})

    print("[3/6] Running 30-trial SimBlock-calibrated DDoS comparison (all 7 models)...")
    models = [
        ("Standard PoW", "Standard PoW (No Defense)"),
        ("ML-IDS", "ML-Based IDS (Network Layer)"),
        ("Static TOPSIS", "Static TOPSIS (Pre-processing)"),
        ("EWMA Reputation", "EWMA Reputation Filter"),
        ("Token Bucket", "Rate Limiting (Token Bucket)"),
        ("Dynamic Fee", "Dynamic Fee Mechanism"),
        ("A-OBCTrm-PoW", "A-OBCTrm-PoW (Proposed)"),
    ]
    ddos_trials: dict[str, list[dict]] = {}
    nominal_trials: dict[str, list[dict]] = {}
    all_rows_csv = []

    for short_name, full_label in models:
        ddos_trials[short_name] = []
        for s in seeds:
            r = simulate_single_trial(short_name, s, cfg, traitor_fraction=0.30, attack_mode="ddos", wan_mode=False, ml_detector=ml)
            ddos_trials[short_name].append(r)
            all_rows_csv.append({
                "experiment": "ddos_30pct",
                "model": full_label,
                "seed": s,
                "traitor_fraction": 0.30,
                "b_cr_s": round(r["b_cr_mean"], 4),
                "b_prop_s": round(r["b_prop_mean"], 4),
                "fork_rate_pct": round(r["fork_rate_pct"], 4),
                "stale_rate_pct": round(r["stale_rate_pct"], 4),
                "conf_latency_s": round(r["conf_latency_s"], 4),
                "bandwidth_mb_s": round(r["bandwidth_mb_s"], 4),
                "honest_penalty_pct": round(r["honest_penalty_pct"], 4),
                "fpr_pct": round(r["fpr_pct"], 4),
                "fnr_pct": round(r["fnr_pct"], 4),
                "filtered_pct": round(r["filtered_pct"], 4),
                "pool_uptime_pct": round(r["pool_uptime_pct"], 4),
            })

        if short_name in ("A-OBCTrm-PoW", "Static TOPSIS", "ML-IDS", "Standard PoW"):
            nominal_trials[short_name] = [
                simulate_single_trial(short_name, s, cfg, traitor_fraction=0.10, attack_mode="nominal", wan_mode=False, ml_detector=ml)
                for s in seeds
            ]

    print("[4/6] Running adversarial scaling (10%, 30%, 50%, corrupted-feedback 50%, Sybil, multi-vector)...")
    adversarial_regimes = {}
    scenarios = [
        ("Low (10% traitors)", "A-OBCTrm-PoW", 0.10, "ddos"),
        ("Medium (30% traitors)", "A-OBCTrm-PoW", 0.30, "ddos"),
        ("High (50% traitors, reliable feedback)", "A-OBCTrm-PoW", 0.50, "ddos"),
        ("High (50% traitors, corrupted feedback)", "A-OBCTrm-PoW", 0.50, "corrupted_feedback"),
        ("High (50% traitors, sleeper feedback poisoning)", "A-OBCTrm-PoW", 0.50, "sleeper_feedback"),
        ("Feedback Collusion (30% Sybil, A-OBCTrm-PoW)", "A-OBCTrm-PoW", 0.30, "sybil"),
        ("Feedback Collusion (30% Sybil, Uniform PRF)", "Static TOPSIS", 0.30, "sybil"),
        ("Combined Multi-Vector (DDoS + Sybil + Collusion)", "A-OBCTrm-PoW", 0.30, "multivector"),
    ]
    for label, mod, frac, mode in scenarios:
        runs = [simulate_single_trial(mod, s, cfg, traitor_fraction=frac, attack_mode=mode, wan_mode=False, ml_detector=ml) for s in seeds]
        feff_vals = [
            (frac * (r["fnr_pct"] / 100.0))
            / max(1e-12, (1.0 - frac) * (1.0 - r["fpr_pct"] / 100.0) + frac * (r["fnr_pct"] / 100.0))
            for r in runs
        ]
        adversarial_regimes[label] = {
            "filtered_pct": agg_metric([r["filtered_pct"] for r in runs]),
            "fnr_pct": agg_metric([r["fnr_pct"] for r in runs]),
            "fpr_pct": agg_metric([r["fpr_pct"] for r in runs]),
            "f_eff": agg_metric(feff_vals),
            "pool_uptime_pct": agg_metric([r["pool_uptime_pct"] for r in runs]),
            "hm_pool_pct": agg_metric([r["hm_pool_pct"] for r in runs]),
            "conf_latency_s": agg_metric([r["conf_latency_s"] for r in runs]),
            "stale_rate_pct": agg_metric([r["stale_rate_pct"] for r in runs]),
            "b_cr_s": agg_metric([r["b_cr_mean"] for r in runs]),
        }

    # Nominal confirmation latency baseline (undefended 10% nominal) to measure exact +delta%
    nom_conf_vals = [r["conf_latency_s"] for r in nominal_trials["Standard PoW"]]
    nom_conf_mean = float(np.mean(nom_conf_vals))

    print("[5/6] Running 50-node Log-Normal WAN Emulation & Real CPU Benchmark...")
    wan_results = {}
    for short_name in ("Static TOPSIS", "ML-IDS", "A-OBCTrm-PoW"):
        wan_runs = [simulate_single_trial(short_name, s, cfg, traitor_fraction=0.30, attack_mode="ddos", wan_mode=True, ml_detector=ml) for s in seeds]
        wan_results[short_name] = {
            "b_cr_s": agg_metric([r["b_cr_mean"] for r in wan_runs]),
            "b_prop_s": agg_metric([r["b_prop_mean"] for r in wan_runs]),
        }

    cpu_results = benchmark_model_cpu_and_runtime(cfg, n_rounds=25, n_trials=20)

    print("[6/6] Computing paired statistical tests, KS statistics, and exporting telemetry...")
    prop_bcr = [r["b_cr_mean"] for r in ddos_trials["A-OBCTrm-PoW"]]
    prop_bprop = [r["b_prop_mean"] for r in ddos_trials["A-OBCTrm-PoW"]]
    stats_tests = {
        "b_cr_vs_Static_TOPSIS": paired_stats(prop_bcr, [r["b_cr_mean"] for r in ddos_trials["Static TOPSIS"]]),
        "b_cr_vs_ML_IDS": paired_stats(prop_bcr, [r["b_cr_mean"] for r in ddos_trials["ML-IDS"]]),
        "b_cr_vs_EWMA": paired_stats(prop_bcr, [r["b_cr_mean"] for r in ddos_trials["EWMA Reputation"]]),
        "b_cr_vs_Standard_PoW": paired_stats(prop_bcr, [r["b_cr_mean"] for r in ddos_trials["Standard PoW"]]),
        "b_prop_vs_Static_TOPSIS": paired_stats(prop_bprop, [r["b_prop_mean"] for r in ddos_trials["Static TOPSIS"]]),
        "b_prop_vs_Standard_PoW": paired_stats(prop_bprop, [r["b_prop_mean"] for r in ddos_trials["Standard PoW"]]),
    }

    # Trust score distribution & 2-sample KS test from trial seed 100 of A-OBCTrm-PoW
    sample_r = ddos_trials["A-OBCTrm-PoW"][0]
    c_arr = np.asarray(sample_r["final_closeness"], dtype=np.float64)
    m_arr = np.asarray(sample_r["final_is_mal"], dtype=bool)
    ks_stat, ks_p = stats.ks_2samp(c_arr[~m_arr], c_arr[m_arr])

    # Aggregate baseline comparison & TPS series
    baseline_summary = {}
    tps_curves = {}
    for short_name, full_label in models:
        runs = ddos_trials[short_name]
        baseline_summary[full_label] = {
            "b_cr_s": agg_metric([r["b_cr_mean"] for r in runs]),
            "b_prop_s": agg_metric([r["b_prop_mean"] for r in runs]),
            "fork_rate_pct": agg_metric([r["fork_rate_pct"] for r in runs]),
            "bandwidth_mb_s": agg_metric([r["bandwidth_mb_s"] for r in runs]),
            "honest_penalty_pct": agg_metric([r["honest_penalty_pct"] for r in runs]),
            "cpu_overhead_pct": {
                "mean": round(cpu_results[full_label]["cpu_overhead_pct_mean"], 2),
                "sd": round(cpu_results[full_label]["cpu_overhead_pct_sd"], 2),
                "per_round_ms": round(cpu_results[full_label]["per_round_ms_mean"], 2),
            },
        }
        tps_mat = np.asarray([r["tps_series"] for r in runs], dtype=np.float64)
        tps_curves[short_name] = {
            "mean": np.round(tps_mat.mean(axis=0), 2).tolist(),
            "sd": np.round(tps_mat.std(axis=0, ddof=1), 2).tolist(),
        }

    nominal_summary = {
        k: {"b_cr_s": agg_metric([r["b_cr_mean"] for r in v]), "b_prop_s": agg_metric([r["b_prop_mean"] for r in v])}
        for k, v in nominal_trials.items()
    }

    master_output = {
        "metadata": {
            "num_miners": cfg.num_miners,
            "num_trials": cfg.num_trials,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "cpu_hardware": cpu_results.get("cpu_hardware", "Unknown"),
            "slot_cpu_budget_ms": cpu_results.get("slot_cpu_budget_ms", 92.0),
            "elapsed_wall_s": round(time.perf_counter() - t_start, 2),
        },
        "sample_pow_blocks": pow_blocks,
        "nominal_summary": nominal_summary,
        "nominal_undefended_conf_latency_s": round(nom_conf_mean, 4),
        "baseline_ddos_comparison": baseline_summary,
        "adversarial_regimes": adversarial_regimes,
        "wan_50_node_validation": wan_results,
        "statistical_tests": stats_tests,
        "trust_distribution_sample": {
            "threshold_T": round(sample_r["final_threshold"], 4),
            "ks_D": round(float(ks_stat), 4),
            "ks_p_value": float(ks_p),
            "honest_scores": np.round(c_arr[~m_arr], 4).tolist(),
            "malicious_scores": np.round(c_arr[m_arr], 4).tolist(),
        },
        "cpu_benchmark_details": cpu_results,
        "tps_curves": tps_curves,
    }

    with open(RESULTS_DIR / "all_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(master_output, f, indent=2)

    # Re-run proof verification against newly generated master_output so Theorem 4 telemetry is synchronized
    run_proofs(exp_results=master_output)

    with open(RESULTS_DIR / "trial_telemetry.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows_csv[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows_csv)

    print(f"Completed all 30-trial experiments in {time.perf_counter() - t_start:.2f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
