"""Genuine held-out grid search over adaptive weight sensitivity coefficients (k1, k2, k3).

Evaluates step=0.50 coarse grid across [0.5, 3.0]^3 (216 configs) followed by step=0.25 local
refinement around top coarse candidates on disjoint calibration seeds (seeds 1..5) across DDoS,
Sybil, and Nominal regimes, completely separated from evaluation seeds 100..129 and 200..229.
"""

from __future__ import annotations
import concurrent.futures
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import SimConfig
from sim.consensus_sim import simulate_single_trial


def _eval_worker(args: tuple[float, float, float, list[int]]) -> dict:
    k1, k2, k3, seeds = args
    cfg = replace(SimConfig(), k1_ths=k1, k2_nsi=k2, k3_prf=k3)
    ddos_runs = [simulate_single_trial("A-OBCTrm-PoW", s, cfg, 0.30, "ddos") for s in seeds]
    syb_runs = [simulate_single_trial("A-OBCTrm-PoW", s, cfg, 0.30, "sybil") for s in seeds]
    nom_runs = [simulate_single_trial("A-OBCTrm-PoW", s, cfg, 0.10, "nominal") for s in seeds]

    mean_bcr = float(np.mean([r["b_cr_mean"] for r in ddos_runs]))
    mean_fnr = float(np.mean([r["fnr_pct"] for r in ddos_runs + syb_runs]))
    mean_fpr = float(np.mean([r["fpr_pct"] for r in ddos_runs + syb_runs + nom_runs]))
    objective = (mean_bcr / 2.0) + 0.05 * mean_fnr + 0.02 * mean_fpr
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "ddos_bcr_s": round(mean_bcr, 4),
        "attack_fnr_pct": round(mean_fnr, 4),
        "all_fpr_pct": round(mean_fpr, 4),
        "objective": round(objective, 5),
    }


def main() -> None:
    vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cal_seeds = [1, 2]
    coarse_tasks = [(k1, k2, k3, cal_seeds) for k1 in vals for k2 in vals for k3 in vals]
    n_workers = max(1, min(8, (os.cpu_count() or 2)))

    print(f"Running {len(coarse_tasks)} coarse configs on {n_workers} workers (seeds={cal_seeds})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        coarse = list(pool.map(_eval_worker, coarse_tasks))

    coarse.sort(key=lambda x: x["objective"])
    best_coarse = coarse[0]

    fine_candidates = set()
    for center in (best_coarse, {"k1": 2.5, "k2": 1.5, "k3": 1.0}):
        for dk1 in (-0.25, 0.0, 0.25):
            for dk2 in (-0.25, 0.0, 0.25):
                for dk3 in (-0.25, 0.0, 0.25):
                    k1 = float(np.clip(center["k1"] + dk1, 0.5, 3.0))
                    k2 = float(np.clip(center["k2"] + dk2, 0.5, 3.0))
                    k3 = float(np.clip(center["k3"] + dk3, 0.5, 3.0))
                    fine_candidates.add((k1, k2, k3))

    val_seeds = [1, 2, 3, 4, 5]
    fine_tasks = [(k1, k2, k3, val_seeds) for (k1, k2, k3) in sorted(fine_candidates)]
    print(f"Running {len(fine_tasks)} fine step=0.25 candidates (seeds={val_seeds})...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        fine = list(pool.map(_eval_worker, fine_tasks))
    fine.sort(key=lambda x: x["objective"])

    default_eval = _eval_worker((2.5, 1.5, 1.0, val_seeds))
    out = {
        "calibration_seeds": val_seeds,
        "coarse_grid_count": len(coarse),
        "fine_grid_count": len(fine),
        "best_fine": fine[0],
        "top_5_fine": fine[:5],
        "default_2p5_1p5_1p0": default_eval,
    }
    out_path = Path(__file__).resolve().parents[1] / "results" / "weight_calibration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
