"""Real CPU utilization and execution latency benchmark comparing consensus filtering models.

Measures shared baseline PoW/SHA-256 verification cost alongside each model's exact
incremental computation cost (feature extraction + RF inference vs P2P TOPSIS ranking)
to prevent OS scheduling jitter from inverting microsecond-level differences.
"""

from __future__ import annotations
import hashlib
import platform
import time
import numpy as np
from .config import SimConfig
from .trust_engine import (
    compute_adaptive_weights,
    compute_credibility_prf,
    compute_dynamic_threshold,
    compute_topsis_closeness,
)
from .baselines import (
    MLIDSDetector,
    dynamic_fee_filter,
    ewma_update_and_admit,
    static_topsis_admit,
    token_bucket_filter,
)


def _get_cpu_hardware_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip()
    except Exception:
        return platform.processor() or platform.machine() or "Unknown CPU"


def benchmark_model_cpu_and_runtime(cfg: SimConfig, n_rounds: int = 25, n_trials: int = 20) -> dict:
    """Measure real execution latency and single-vCPU slot utilization across models."""
    rng = np.random.default_rng(123)
    m = cfg.num_miners
    is_mal = np.zeros(m, dtype=bool)
    is_mal[700:] = True
    ths = np.where(is_mal, 0.22, 0.85)
    nsi = np.where(is_mal, 0.30, 0.88)
    k_nbr = cfg.peer_feedback_degree
    nbr_indices = rng.integers(0, m, size=(m, k_nbr))
    nbr_ratings = rng.uniform(0.60, 0.90, size=(m, k_nbr))
    ewma_scores = np.where(is_mal, 0.40, 0.82)
    window_telemetry = rng.normal(0.5, 0.1, size=(50, m))

    detector = MLIDSDetector(seed=42)
    # Pre-allocated transaction payloads for reproducible SHA-256 block/mempool validation
    tx_payloads = [f"tx_payload_{i:05d}".encode("ascii") for i in range(9500)]
    slot_cpu_budget_ms = 92.0

    # 1. Measure baseline shared SHA-256 mempool/block validation work across 20 trials
    base_ms_trials = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_rounds):
            for pld in tx_payloads:
                hashlib.sha256(pld).digest()
        base_ms_trials.append((time.perf_counter() - t0) * 1000.0 / n_rounds)
    base_arr = np.asarray(base_ms_trials, dtype=np.float64)

    # 2. Measure incremental per-slot compute time strictly required by each defense mechanism
    models = [
        "Standard PoW (No Defense)",
        "ML-Based IDS (Network Layer)",
        "Static TOPSIS (Pre-processing)",
        "EWMA Reputation Filter",
        "Rate Limiting (Token Bucket)",
        "Dynamic Fee Mechanism",
        "A-OBCTrm-PoW (Proposed)",
    ]

    results = {"cpu_hardware": _get_cpu_hardware_name(), "slot_cpu_budget_ms": slot_cpu_budget_ms}
    for name in models:
        inc_ms_trials = []
        # Run 4x rounds for fast vectorized models to minimize OS timer quantization; ML runs 1x per slot cleanly
        inner_mult = 1 if "ML-Based" in name else 4
        eff_rounds = n_rounds * inner_mult
        for _ in range(n_trials):
            t0 = time.perf_counter()
            for _ in range(eff_rounds):
                if name == "Standard PoW (No Defense)":
                    pass
                elif name == "ML-Based IDS (Network Layer)":
                    detector.extract_and_classify(m, is_mal, True, rng)
                elif name == "Static TOPSIS (Pre-processing)":
                    # Rolling NSI stability aggregation + unweighted peer ratings + fixed TOPSIS
                    _ = window_telemetry.mean(axis=0)
                    r_mean = compute_credibility_prf(ths, nbr_ratings, cfg, uniform_fallback=True, neighbor_indices=nbr_indices)
                    static_topsis_admit(ths, nsi, r_mean)
                elif name == "EWMA Reputation Filter":
                    ewma_update_and_admit(ewma_scores, ths, cfg)
                elif name == "A-OBCTrm-PoW (Proposed)":
                    # Rolling W=50 std telemetry + credibility-weighted PRF + adaptive TOPSIS + trimmed threshold
                    _ = window_telemetry.std(axis=0)
                    w = compute_adaptive_weights(0.65, 0.85, 0.70, cfg)
                    prf = compute_credibility_prf(ths, nbr_ratings, cfg, uniform_fallback=False, neighbor_indices=nbr_indices)
                    c = compute_topsis_closeness(ths, nsi, prf, w)
                    compute_dynamic_threshold(c, 0.70, cfg)
                elif name == "Rate Limiting (Token Bucket)":
                    token_bucket_filter(
                        cfg.nominal_tx_rate,
                        cfg.ddos_attack_rate,
                        cfg.token_bucket_capacity,
                        cfg.token_bucket_rate,
                        cfg.token_bucket_peers,
                    )
                elif name == "Dynamic Fee Mechanism":
                    dynamic_fee_filter(cfg.nominal_tx_rate, cfg.ddos_attack_rate, 0.85, cfg)
            inc_ms_trials.append((time.perf_counter() - t0) * 1000.0 / eff_rounds)

        total_ms = base_arr + np.asarray(inc_ms_trials, dtype=np.float64)
        cpu_pct = np.minimum(98.0, (total_ms / slot_cpu_budget_ms) * 100.0)
        results[name] = {
            "per_round_ms_mean": float(np.mean(total_ms)),
            "per_round_ms_sd": float(np.std(total_ms, ddof=1)),
            "incremental_ms_mean": float(np.mean(inc_ms_trials)),
            "cpu_overhead_pct_mean": float(np.mean(cpu_pct)),
            "cpu_overhead_pct_sd": float(np.std(cpu_pct, ddof=1)),
        }

    # Measure isolated O(m*n) TOPSIS + threshold update latency (excluding rolling window telemetry read)
    t0 = time.perf_counter()
    for _ in range(500):
        w = compute_adaptive_weights(0.65, 0.85, 0.70, cfg)
        D = np.column_stack([ths, nsi, ths])
        R = D / (np.linalg.norm(D, axis=0) + 1e-12)
        V = R * w
        Ap, An = V.max(axis=0), V.min(axis=0)
        Sp = np.linalg.norm(V - Ap, axis=1)
        Sn = np.linalg.norm(V - An, axis=1)
        C = Sn / (Sp + Sn + 1e-12)
        compute_dynamic_threshold(C, 0.70, cfg)
    results["topsis_kernel_only_ms"] = float((time.perf_counter() - t0) * 1000.0 / 500.0)
    return results
