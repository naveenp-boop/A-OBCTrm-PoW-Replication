"""Print concise verification summary of all simulation metrics matching paper Tables 3-8 & Figs 2-5."""
import json
from pathlib import Path

r_dir = Path(__file__).resolve().parents[1] / "results"
exp = json.loads((r_dir / "all_experiment_results.json").read_text(encoding="utf-8"))
proofs = json.loads((r_dir / "proof_verification.json").read_text(encoding="utf-8"))

print("=== METADATA ===")
print(exp["metadata"])

print("\n=== SAMPLE MINED SHA-256 16-BIT POW BLOCKS ===")
for b in exp["sample_pow_blocks"]:
    print(b)

print("\n=== NOMINAL SUMMARY ===")
for k, v in exp["nominal_summary"].items():
    print(f"  {k:16s}: Bcr={v['b_cr_s']['mean']:.2f}±{v['b_cr_s']['sd']:.2f}s, Bprop={v['b_prop_s']['mean']:.2f}±{v['b_prop_s']['sd']:.2f}s")

print("\n=== TABLE 3 & TABLE 7: BASELINE DDOS COMPARISON (30% TRAITORS, 5000 tx/s) ===")
for k, v in exp["baseline_ddos_comparison"].items():
    print(f"  {k:32s} | Bcr={v['b_cr_s']['mean']:.2f}±{v['b_cr_s']['sd']:.2f} | Bprop={v['b_prop_s']['mean']:.2f}±{v['b_prop_s']['sd']:.2f} | CPU={v['cpu_overhead_pct']['mean']:.1f}±{v['cpu_overhead_pct']['sd']:.1f}% ({v['cpu_overhead_pct']['per_round_ms']}ms) | Pen={v['honest_penalty_pct']['mean']:.1f}% | Fork={v['fork_rate_pct']['mean']:.2f}±{v['fork_rate_pct']['sd']:.2f}% | BW={v['bandwidth_mb_s']['mean']:.1f}±{v['bandwidth_mb_s']['sd']:.1f} MB/s")

print("\n=== STATISTICAL TESTS (n=30) ===")
for k, v in exp["statistical_tests"].items():
    print(f"  {k:28s}: {v}")

print("\n=== ADVERSARIAL REGIMES (TABLE 5 & TABLE 8 & FPR/FNR) ===")
nom_conf = exp["nominal_undefended_conf_latency_s"]
print(f"  Nominal undefended conf latency = {nom_conf:.2f}s")
for k, v in exp["adversarial_regimes"].items():
    delta_conf = (v['conf_latency_s']['mean'] - nom_conf) / nom_conf * 100.0
    print(f"  {k:46s} | HM={v['hm_pool_pct']['mean']:.1f}% | Filt={v['filtered_pct']['mean']:.1f}% | FPR={v['fpr_pct']['mean']:.2f}% | FNR={v['fnr_pct']['mean']:.2f}% | Up={v['pool_uptime_pct']['mean']:.1f}% | Conf={v['conf_latency_s']['mean']:.2f}s ({delta_conf:+.1f}%) | Stale={v['stale_rate_pct']['mean']:.2f}%")

print("\n=== 50-NODE WAN VALIDATION (TABLE 6) ===")
for k, v in exp["wan_50_node_validation"].items():
    print(f"  WAN {k:16s}: Bcr={v['b_cr_s']['mean']:.2f}±{v['b_cr_s']['sd']:.2f}s, Bprop={v['b_prop_s']['mean']:.2f}±{v['b_prop_s']['sd']:.2f}s")

print("\n=== TRUST DIST / KS / TOPSIS KERNEL / TPS ENDPOINTS ===")
td = exp["trust_distribution_sample"]
print(f"  T={td['threshold_T']}, KS D={td['ks_D']}, p={td['ks_p_value']:.2e}")
print(f"  TOPSIS kernel only (ms): {exp['cpu_benchmark_details']['topsis_kernel_only_ms']:.3f}")
for m in ["A-OBCTrm-PoW", "ML-IDS", "Static TOPSIS", "Standard PoW"]:
    s = exp["tps_curves"][m]["mean"]
    print(f"  TPS {m:15s}: t0={s[0]:.0f}, t30={s[30]:.0f}, t40={s[40]:.0f}, t80={s[80]:.0f}, t120={s[120]:.0f}")

print("\n=== PROOFS SUMMARY ===")
print(json.dumps(proofs, indent=2))
