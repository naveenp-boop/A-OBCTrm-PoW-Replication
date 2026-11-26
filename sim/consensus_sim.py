"""Discrete-time analytic blockchain consensus & mempool evaluation model.

Executes time-stepped mempool admission and closed-form analytic propagation/decision
evaluations across 1,000 simulated miners (SimBlock-calibrated overlay) and 50 WAN-mode
nodes under DDoS flooding, Sybil cliques, bootstrapping defectors, and multi-vector attacks.
"""

from __future__ import annotations
from collections import deque
import hashlib
import time
import numpy as np
from .config import SimConfig
from .topology import P2PNetworkTopology
from .trust_engine import (
    compute_adaptive_weights,
    compute_credibility_prf,
    compute_dynamic_threshold,
    compute_topsis_closeness,
    update_ths_asymmetric,
)
from .baselines import (
    MLIDSDetector,
    dynamic_fee_filter,
    ewma_update_and_admit,
    static_topsis_admit,
    token_bucket_filter,
)


def solve_scaled_pow_sample(block_header: str, leading_zero_bits: int = 16) -> tuple[int, str, float]:
    """Execute genuine SHA-256 hash puzzle for cryptographic verification in block logs."""
    t0 = time.perf_counter()
    target_prefix_bytes = leading_zero_bits // 8
    rem_bits = leading_zero_bits % 8
    nonce = 0
    zero_bytes = b"\x00" * target_prefix_bytes
    while True:
        digest = hashlib.sha256(f"{block_header}:{nonce}".encode("ascii")).digest()
        if digest[:target_prefix_bytes] == zero_bytes:
            if rem_bits == 0 or (digest[target_prefix_bytes] >> (8 - rem_bits)) == 0:
                return nonce, digest.hex(), (time.perf_counter() - t0) * 1000.0
        nonce += 1


def simulate_single_trial(
    model_name: str,
    seed: int,
    cfg: SimConfig,
    traitor_fraction: float = 0.30,
    attack_mode: str = "ddos",  # "nominal", "ddos", "sybil", "multivector", "corrupted_feedback", "sleeper_feedback"
    wan_mode: bool = False,
    ml_detector: MLIDSDetector | None = None,
) -> dict:
    """Run one complete 120s simulation trial for a given defense model and attack profile."""
    rng = np.random.default_rng(seed)
    n_nodes = cfg.wan_nodes if wan_mode else cfg.num_miners
    topo = P2PNetworkTopology(
        num_nodes=n_nodes,
        rng=rng,
        wan_mode=wan_mode,
        mu_ln=cfg.wan_mu_ln,
        sigma_ln=cfg.wan_sigma_ln,
        packet_loss=cfg.wan_packet_loss,
        backbone_bw_mbps=cfg.backbone_bw_mbps,
        edge_bw_mbps=cfg.edge_bw_mbps,
    )

    n_mal = int(round(n_nodes * traitor_fraction))
    is_mal = np.zeros(n_nodes, dtype=bool)
    is_mal[n_nodes - n_mal :] = True
    rng.shuffle(is_mal)

    # Realistic steady-state node trust distributions with slight tail overlap
    ths = np.where(is_mal, rng.normal(0.30, 0.11, n_nodes), rng.normal(0.80, 0.08, n_nodes))
    nsi = np.where(is_mal, rng.normal(0.38, 0.12, n_nodes), rng.normal(0.82, 0.08, n_nodes))
    ths = np.clip(ths, 0.02, 0.98)
    nsi = np.clip(nsi, 0.05, 0.99)
    ewma_score = np.where(is_mal, rng.uniform(0.38, 0.62, n_nodes), rng.uniform(0.68, 0.92, n_nodes))

    # Probation state for ~3% newly joined honest miners during run
    new_honest_mask = (~is_mal) & (rng.random(n_nodes) < 0.03)
    probation_left = np.where(new_honest_mask, cfg.n_min_probation, 0)
    ths[new_honest_mask] = cfg.gamma_probation * float(np.median(ths))

    # Build P2P neighborhood feedback matrix N(i) of size K = min(peer_feedback_degree, n_nodes)
    k_nbr = min(cfg.peer_feedback_degree, n_nodes)
    nbr_indices = rng.integers(0, n_nodes, size=(n_nodes, k_nbr))
    rater_mal = is_mal[nbr_indices]  # shape (n_nodes, K)
    target_mal = is_mal[:, None]     # shape (n_nodes, 1)

    nbr_ratings = rng.normal(0.80, 0.08, size=(n_nodes, k_nbr))
    nbr_ratings[(~rater_mal) & target_mal] = rng.normal(0.28, 0.10, size=int(((~rater_mal) & target_mal).sum()))

    if attack_mode in ("sybil", "multivector", "corrupted_feedback", "sleeper_feedback"):
        nsi[is_mal] = np.clip(rng.normal(0.60, 0.12, n_mal), 0.15, 0.92)
        ths[is_mal] = np.clip(rng.normal(0.34, 0.12, n_mal), 0.05, 0.68)
        m_mm = int((rater_mal & target_mal).sum())
        m_mh = int((rater_mal & (~target_mal)).sum())
        if m_mm > 0:
            nbr_ratings[rater_mal & target_mal] = rng.normal(0.95, 0.03, size=m_mm)
        if m_mh > 0:
            nbr_ratings[rater_mal & (~target_mal)] = rng.normal(0.24, 0.10, size=m_mh)
        if attack_mode in ("corrupted_feedback", "sleeper_feedback") and traitor_fraction > 0.333:
            # Aged compromised accounts start with accumulated historical trust before poisoning ratings
            comp_idx = np.where(is_mal)[0][: int(0.85 * n_mal)]
            ths[comp_idx] = rng.uniform(0.64, 0.88, size=len(comp_idx))
    else:
        m_mm = int((rater_mal & target_mal).sum())
        m_mh = int((rater_mal & (~target_mal)).sum())
        if m_mm > 0:
            nbr_ratings[rater_mal & target_mal] = rng.normal(0.58, 0.13, size=m_mm)
        if m_mh > 0:
            nbr_ratings[rater_mal & (~target_mal)] = rng.normal(0.68, 0.10, size=m_mh)
    nbr_ratings = np.clip(nbr_ratings, 0.0, 1.0)

    tps_series = np.zeros(cfg.sim_duration_s + 1, dtype=np.float64)
    bcr_samples = []
    bprop_samples = []
    fork_events = []
    stale_events = []
    conf_lat_samples = []
    bw_samples = []
    fpr_samples = []
    fnr_samples = []
    honest_penalty_samples = []
    pool_uptime_ticks = 0
    mempool_q = 0.0

    last_closeness = None
    last_threshold = 0.52
    tb_tokens = float(cfg.token_bucket_peers * cfg.token_bucket_capacity)

    # Sliding W-tick causal telemetry windows (no ground-truth traitor_fraction or is_mal access)
    bprop_hist: deque[float] = deque([0.42], maxlen=cfg.window_w)
    arrival_hist: deque[float] = deque([cfg.nominal_tx_rate], maxlen=cfg.window_w)
    defect_hist: deque[float] = deque([0.0], maxlen=cfg.window_w)
    # Precompute neighbor-rating dispersion observable from current P2P feedback matrix
    rating_disp = float(np.mean(np.std(nbr_ratings, axis=1)))

    for t in range(cfg.sim_duration_s + 1):
        attack_active = (attack_mode != "nominal") and (t >= cfg.ddos_start_s)
        # Pre-attack baseline carries mild background chatter; DDoS flood strikes at t=30 s
        if not attack_active:
            atk_scale = 0.0 if attack_mode == "nominal" else 0.04
        else:
            atk_scale = 1.0

        honest_in = cfg.nominal_tx_rate
        spam_in = (cfg.ddos_attack_rate * atk_scale) if attack_mode in ("ddos", "multivector", "corrupted_feedback", "sleeper_feedback") else (1800.0 * atk_scale)
        total_arrival = honest_in + spam_in
        arrival_hist.append(total_arrival)

        # Causal observable telemetry drivers over sliding W=50 horizon (zero oracle leakage)
        mean_bprop_w = float(np.mean(bprop_hist))
        obs_bprop_norm = float(np.clip((mean_bprop_w - 0.35) / 1.50, 0.0, 1.0))
        obs_excess_cap = float(np.clip((total_arrival - cfg.service_capacity_tx_s) / cfg.ddos_attack_rate, 0.0, 1.0))
        q_norm_cur = float(np.clip(mempool_q / cfg.mempool_capacity_tx, 0.0, 1.0))
        lam_hat = float(np.clip(0.55 * obs_bprop_norm + 0.45 * obs_excess_cap + 0.25 * q_norm_cur, 0.0, 1.0))

        peak_or_cur_arr = 0.75 * total_arrival + 0.25 * float(np.mean(arrival_hist))
        tau_hat = float(np.clip(peak_or_cur_arr / (cfg.nominal_tx_rate + cfg.ddos_attack_rate), 0.0, 1.0))

        excess_flux = float(np.clip((total_arrival - cfg.nominal_tx_rate) / cfg.ddos_attack_rate, 0.0, 1.0))
        low_ths_frac = float(np.mean(ths < cfg.theta_c))
        obs_defect_rate = float(np.mean(defect_hist))
        delta_hat = float(np.clip(
            0.45 * excess_flux + 0.30 * low_ths_frac + 0.20 * min(1.0, obs_defect_rate * 4.0) + 0.25 * min(1.0, rating_disp / 0.28),
            0.0,
            1.0,
        ))

        ml_filter_s = 0.0
        extra_honest_pen: float | None = None
        spam_leak_factor: float | None = None

        if model_name == "Standard PoW":
            admitted = np.ones(n_nodes, dtype=bool)

        elif model_name == "ML-IDS":
            if ml_detector is None:
                ml_detector = MLIDSDetector(seed=42)
            # Measure actual wall-clock RF inference latency per validation slot
            t_infer0 = time.perf_counter()
            admitted = ml_detector.extract_and_classify(n_nodes, is_mal, attack_active, rng)
            ml_filter_s = time.perf_counter() - t_infer0

        elif model_name == "Static TOPSIS":
            raw_prf = compute_credibility_prf(ths, nbr_ratings, cfg, uniform_fallback=True, neighbor_indices=nbr_indices)
            admitted, _ = static_topsis_admit(ths, nsi, raw_prf)

        elif model_name == "EWMA Reputation":
            round_q = np.where(
                is_mal,
                rng.uniform(0.25 if attack_active else 0.50, 0.72, n_nodes),
                rng.uniform(0.70, 0.96, n_nodes),
            )
            admitted, ewma_score = ewma_update_and_admit(ewma_score, round_q, cfg, threshold=0.58)

        elif model_name == "Token Bucket":
            admitted = np.ones(n_nodes, dtype=bool)
            _, adm_a, tb_pen, tb_tokens = token_bucket_filter(
                honest_in,
                spam_in,
                cfg.token_bucket_capacity,
                cfg.token_bucket_rate,
                num_ingress_peers=cfg.token_bucket_peers,
                bucket_tokens=tb_tokens,
            )
            spam_leak_factor = (adm_a / max(1.0, spam_in)) if spam_in > 0 else 0.0
            extra_honest_pen = tb_pen

        elif model_name == "Dynamic Fee":
            admitted = np.ones(n_nodes, dtype=bool)
            occ = (mempool_q + honest_in + spam_in) / cfg.mempool_capacity_tx
            _, adm_a, fee_pen = dynamic_fee_filter(honest_in, spam_in, occ, cfg)
            spam_leak_factor = (adm_a / max(1.0, spam_in)) if spam_in > 0 else 0.0
            extra_honest_pen = fee_pen

        elif model_name == "A-OBCTrm-PoW":
            weights = compute_adaptive_weights(lam_hat, tau_hat, delta_hat, cfg)
            prf = compute_credibility_prf(ths, nbr_ratings, cfg, uniform_fallback=False, neighbor_indices=nbr_indices)
            closeness = compute_topsis_closeness(ths, nsi, prf, weights, probation_left, cfg)
            threshold, _, _ = compute_dynamic_threshold(closeness, delta_hat, cfg)
            admitted = closeness >= threshold
            last_closeness = closeness
            last_threshold = threshold

            admitted_indices = np.where(admitted)[0]
            proposers = np.zeros(n_nodes, dtype=bool)
            if len(admitted_indices) > 0:
                chosen = rng.choice(
                    admitted_indices,
                    size=min(len(admitted_indices), max(2, int(0.05 * len(admitted_indices)))),
                    replace=False,
                )
                proposers[chosen] = True

            valid_outcomes = np.ones(n_nodes, dtype=bool)
            mal_prop = proposers & is_mal
            # In corrupted_feedback (active defectors), compromised nodes keep uniform defect_prob=0.28.
            # In explicit sleeper_feedback, compromised validators mimic honest block validity (0.004) while poisoning feedback ratings.
            if attack_mode == "sleeper_feedback":
                defect_prob = 0.004
            else:
                defect_prob = 0.28 if attack_active else 0.14
            valid_outcomes[mal_prop] = ~(rng.random(int(mal_prop.sum())) < defect_prob)
            hon_prop = proposers & (~is_mal)
            valid_outcomes[hon_prop] = ~(rng.random(int(hon_prop.sum())) < 0.004)
            defect_hist.append(float((proposers & (~valid_outcomes)).sum()) / max(1, int(proposers.sum())))

            ths = update_ths_asymmetric(ths, valid_outcomes, proposers, cfg)
            nsi = np.clip(nsi + rng.normal(0.0, 0.004, size=n_nodes), 0.05, 0.99)
            probation_left = np.maximum(0, probation_left - (admitted.astype(int)))

        else:
            raise ValueError(f"Unknown model: {model_name}")

        n_hon = int((~is_mal).sum())
        fpr = float((~admitted & ~is_mal).sum()) / max(1, n_hon)
        fnr = float((admitted & is_mal).sum()) / max(1, n_mal)
        fpr_samples.append(fpr)
        fnr_samples.append(fnr)

        # Drive all admission-based filters symmetrically from their own emergent FNR and uniform honest-relay loss weight
        if spam_leak_factor is None:
            spam_leak_factor = fnr
        if extra_honest_pen is None:
            extra_honest_pen = fpr * cfg.honest_relay_penalty_weight

        admitted_spam = spam_in * spam_leak_factor
        admitted_honest = honest_in * (1.0 - extra_honest_pen)
        total_ingress = admitted_honest + admitted_spam

        mal_share_admitted = float((admitted & is_mal).sum()) / max(1, int(admitted.sum()))
        # Service capacity clears 2850 tx/s cleanly when pre-attack or clean, slowing smoothly under equivocation/ML lag
        service_rate = cfg.service_capacity_tx_s * (1.0 - 0.42 * mal_share_admitted) / (1.0 + 0.15 * ml_filter_s)
        net_delta = total_ingress - service_rate
        mempool_q = float(np.clip(mempool_q + net_delta, 0.0, cfg.mempool_capacity_tx * 1.3))
        q_norm = mempool_q / cfg.mempool_capacity_tx
        load_ratio = float(np.clip(q_norm + admitted_spam / cfg.service_capacity_tx_s, 0.0, 2.0))

        # Goodput TPS bounded by both honest arrival and effective non-congested service bandwidth
        effective_honest_share = admitted_honest / max(1.0, total_ingress + 0.22 * mempool_q)
        saturation_penalty = 1.0 / (1.0 + 2.8 * (max(0.0, q_norm - 0.35) ** 2.1))
        raw_goodput = min(admitted_honest, service_rate * effective_honest_share) * saturation_penalty
        goodput = float(np.clip(raw_goodput + rng.normal(0, 14.0), 45.0, cfg.nominal_tx_rate))
        tps_series[t] = goodput

        b_prop = topo.compute_propagation_delay(
            admitted_mask=admitted,
            is_malicious=is_mal,
            mempool_load_ratio=load_ratio,
            filter_latency_s=0.55 * ml_filter_s,
            block_size_mb=cfg.block_size_mb,
        )
        bprop_hist.append(b_prop)

        fork_rate = float(np.clip(0.004 + 0.015 * b_prop + 0.08 * mal_share_admitted, 0.002, 0.22))
        stale_rate = float(np.clip(0.42 * fork_rate + 0.045 * mal_share_admitted, 0.001, 0.18))

        base_decision_s = 1.65
        congestion_bcr = 0.45 * load_ratio + 1.35 * (load_ratio ** 1.15)
        fork_inflation = 1.0 / max(0.38, 1.0 - 2.0 * fork_rate)
        b_cr = (base_decision_s + congestion_bcr + 1.90 * ml_filter_s + 0.85 * b_prop) * fork_inflation + float(rng.normal(0, 0.07))
        b_cr = max(1.4, b_cr)

        hon_pool_frac = float((admitted & ~is_mal).sum()) / max(1, n_nodes)
        conf_lat = b_cr * (1.85 + 0.55 * max(0.0, 0.90 - hon_pool_frac))
        bw_mb_s = (admitted_honest + admitted_spam * 1.4) * (cfg.tx_size_bytes / 1e6) * (3.8 + 4.5 * fork_rate) + (cfg.block_size_mb * (1.0 + fork_rate) * 6.5)

        if t >= cfg.ddos_start_s or attack_mode == "nominal":
            bcr_samples.append(b_cr)
            bprop_samples.append(b_prop)
            fork_events.append(fork_rate * 100.0)
            stale_events.append(stale_rate * 100.0)
            conf_lat_samples.append(conf_lat)
            bw_samples.append(bw_mb_s)
            honest_penalty_samples.append(extra_honest_pen * 100.0)
            if (1.0 - mal_share_admitted) > (2.0 / 3.0) and goodput >= 0.50 * cfg.nominal_tx_rate:
                pool_uptime_ticks += 1

    active_ticks = len(bcr_samples)
    return {
        "model": model_name,
        "seed": seed,
        "traitor_fraction": traitor_fraction,
        "attack_mode": attack_mode,
        "wan_mode": wan_mode,
        "b_cr_mean": float(np.mean(bcr_samples)),
        "b_prop_mean": float(np.mean(bprop_samples)),
        "fork_rate_pct": float(np.mean(fork_events)),
        "stale_rate_pct": float(np.mean(stale_events)),
        "conf_latency_s": float(np.mean(conf_lat_samples)),
        "bandwidth_mb_s": float(np.mean(bw_samples)),
        "honest_penalty_pct": float(np.mean(honest_penalty_samples)),
        "fpr_pct": float(np.mean(fpr_samples[cfg.ddos_start_s:] if attack_mode != "nominal" else fpr_samples) * 100.0),
        "fnr_pct": float(np.mean(fnr_samples[cfg.ddos_start_s:] if attack_mode != "nominal" else fnr_samples) * 100.0),
        "filtered_pct": float((1.0 - np.mean(fnr_samples[cfg.ddos_start_s:] if attack_mode != "nominal" else fnr_samples)) * 100.0),
        "pool_uptime_pct": float(100.0 * pool_uptime_ticks / max(1, active_ticks)),
        "hm_pool_pct": float((1.0 - traitor_fraction) * (1.0 - np.mean(fpr_samples[-10:])) * 100.0 + traitor_fraction * np.mean(fnr_samples[-10:]) * 100.0),
        "tps_series": tps_series.tolist(),
        "final_closeness": last_closeness.tolist() if last_closeness is not None else None,
        "final_is_mal": is_mal.astype(int).tolist(),
        "final_threshold": last_threshold,
    }
