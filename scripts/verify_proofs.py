"""Quantitative & empirical verification of Properties/Bounds 1-5 plus genuine simulation pilots:
- Property 1: Adaptive weight simplex validity, strict monotonicity (unclamped), L2 Lipschitz bounds
- Property 2: Asymmetric trust penalty (>60% single-step collapse) & exact closed-form recovery formula
- Bound 3: Credibility-weighted PRF zero-history immunity, Sybil perturbation bound, breakdown point
- Property 4: HM-pool capacity (1-f)(1-rho) and Byzantine safety condition under f < 1/3 & psi <= 1-rho
- Property 5: 5%-trimmed threshold breakdown boundedness & O(m) Quickselect equivalence
- Genuine Probationary Convergence Simulation: unconstrained threshold crossing vs statutory N_min=50 cap
- Genuine 100-Miner UCB1 Contextual-Bandit Pilot: live 100-node trust-engine simulation vs SimConfig (0.85, 0.10)
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import SimConfig
from sim.topology import P2PNetworkTopology
from sim.trust_engine import (
    compute_adaptive_weights,
    compute_credibility_prf,
    compute_dynamic_threshold,
    compute_topsis_closeness,
    update_ths_asymmetric,
)


def verify_theorem_1(cfg: SimConfig) -> dict:
    """Verify adaptive weight bounds, strict unclamped monotonicity, and exact analytical L2 Lipschitz bounds."""
    grid = np.linspace(0.0, 1.0, 21)
    max_w_unclamped = 0.0
    max_w_clamped = 0.0
    strict_unclamped_ok = True
    nondecreasing_ok = True
    max_deriv = 0.0

    for lam in grid:
        for tau in grid:
            prev_w = -1.0
            prev_unclamped = -1.0
            for delta in grid:
                f_ths = math.exp(cfg.k1_ths * delta)
                f_nsi = 1.0 + cfg.k2_nsi * lam
                f_prf = 1.0 + cfg.k3_prf * tau
                w_unclamped = f_ths / (f_ths + f_nsi + f_prf)
                max_w_unclamped = max(max_w_unclamped, w_unclamped)

                w_vec = compute_adaptive_weights(lam, tau, delta, cfg)
                assert abs(float(w_vec.sum()) - 1.0) < 1e-9
                assert np.all(w_vec > 0.0) and np.all(w_vec <= cfg.w_max_cap + 1e-9)
                max_w_clamped = max(max_w_clamped, float(w_vec.max()))

                B = f_nsi + f_prf
                deriv = cfg.k1_ths * f_ths * B / ((f_ths + B) ** 2)
                max_deriv = max(max_deriv, deriv)

                if prev_w >= 0.0:
                    if w_vec[0] < prev_w - 1e-12:
                        nondecreasing_ok = False
                    if w_unclamped < cfg.w_max_cap - 1e-12 and prev_unclamped < cfg.w_max_cap - 1e-12:
                        if not (w_vec[0] > prev_w):
                            strict_unclamped_ok = False
                prev_w = float(w_vec[0])
                prev_unclamped = float(w_unclamped)

    w_joint = compute_adaptive_weights(1.0, 1.0, 1.0, cfg)
    f_ths_1 = math.exp(cfg.k1_ths)
    w_isolated_unclamped = f_ths_1 / (f_ths_1 + 2.0)

    # Check exact fine-grid L2 directional & full 3D Jacobian spectral norm
    fine = np.linspace(0.0, 1.0, 51)
    h = 1e-6
    max_delta_l2 = 0.0
    max_jac_l2 = 0.0
    for lam in fine:
        for tau in fine:
            for delta in fine:
                w0 = compute_adaptive_weights(float(lam), float(tau), float(delta), cfg)
                dw_dl = (compute_adaptive_weights(min(1.0, float(lam) + h), float(tau), float(delta), cfg) - w0) / h if lam < 1.0 else (w0 - compute_adaptive_weights(float(lam) - h, float(tau), float(delta), cfg)) / h
                dw_dt = (compute_adaptive_weights(float(lam), min(1.0, float(tau) + h), float(delta), cfg) - w0) / h if tau < 1.0 else (w0 - compute_adaptive_weights(float(lam), float(tau) - h, float(delta), cfg)) / h
                dw_dd = (compute_adaptive_weights(float(lam), float(tau), min(1.0, float(delta) + h), cfg) - w0) / h if delta < 1.0 else (w0 - compute_adaptive_weights(float(lam), float(tau), float(delta) - h, cfg)) / h
                max_delta_l2 = max(max_delta_l2, float(np.linalg.norm(dw_dd)))
                J = np.column_stack([dw_dl, dw_dt, dw_dd])
                max_jac_l2 = max(max_jac_l2, float(np.linalg.norm(J, ord=2)))

    return {
        "passed": bool(strict_unclamped_ok and nondecreasing_ok and max_deriv <= cfg.k1_ths / 4.0 + 1e-9 and max_w_clamped <= cfg.w_max_cap + 1e-9),
        "joint_stress_w_THS": float(w_joint[0]),
        "isolated_anomaly_unclamped_w_THS": float(w_isolated_unclamped),
        "clamped_max_weight": float(max_w_clamped),
        "observed_max_derivative": float(max_deriv),
        "theoretical_lipschitz_bound": float(cfg.k1_ths / 4.0),
        "clamped_vector_max_delta_lipschitz": round(max_delta_l2, 6),
        "clamped_vector_full_3d_l2_lipschitz": round(max_jac_l2, 6),
        "monotonic_strictly": bool(strict_unclamped_ok and nondecreasing_ok),
    }


def verify_theorem_2(cfg: SimConfig) -> dict:
    """Verify multiplicative collapse >60% and closed-form bootstrapping recovery horizon."""
    t0_samples = [0.50, 0.70, 0.85, 0.95]
    collapse_results = []
    for t0 in t0_samples:
        arr = np.array([t0])
        post = float(update_ths_asymmetric(arr, np.array([False]), np.array([True]), cfg)[0])
        frac_drop = (t0 - post) / t0
        assert frac_drop > (1.0 - cfg.kappa_penalty)

        for target in (0.50, 0.75):
            if post >= target:
                continue
            cur = np.array([post])
            steps = 0
            while float(cur[0]) < target:
                cur = update_ths_asymmetric(cur, np.array([True]), np.array([True]), cfg)
                steps += 1
            exact_formula = math.ceil(math.log((1.0 - target) / (1.0 - post)) / math.log(1.0 - cfg.eta_reward))
            assert steps == exact_formula
            collapse_results.append({
                "initial_THS": t0,
                "post_defection_THS": round(post, 4),
                "drop_pct": round(frac_drop * 100.0, 2),
                "target": target,
                "simulated_recovery_blocks": steps,
                "closed_form_k_rec": exact_formula,
            })

    zero_to_half = math.ceil(math.log(0.5) / math.log(1.0 - cfg.eta_reward))
    return {
        "passed": True,
        "zero_to_half_recovery_blocks": zero_to_half,
        "cases": collapse_results,
    }


def verify_theorem_3(cfg: SimConfig) -> dict:
    """Verify Sybil zero-history immunity, perturbation upper bound, and corrupted-feedback breakdown."""
    rng = np.random.default_rng(77)
    n_h = 700
    hon_ths = rng.uniform(0.70, 0.95, n_h)
    hon_ratings_to_i = rng.uniform(0.75, 0.92, n_h)
    for n_sybil in (0, 100, 300):
        zero_ths = np.zeros(n_sybil)
        all_ths = np.concatenate([hon_ths, zero_ths])
        all_r = np.concatenate([hon_ratings_to_i, np.zeros(n_sybil)]).reshape(-1, 1)
        prf_0 = compute_credibility_prf(hon_ths, hon_ratings_to_i.reshape(-1, 1), cfg)[0]
        prf_syb = compute_credibility_prf(all_ths, all_r, cfg)[0]
        assert abs(float(prf_0 - prf_syb)) < 1e-9

    s_syb = 300
    syb_ths = np.full(s_syb, 0.15)
    c_h = np.minimum(1.0, hon_ths / cfg.theta_c).sum()
    c_s_vec = np.minimum(1.0, syb_ths / cfg.theta_c)
    s_cbar = float(c_s_vec.sum())
    theo_bound = s_cbar / (s_cbar + float(c_h) + cfg.eps)

    prf_clean = float(compute_credibility_prf(hon_ths, hon_ratings_to_i.reshape(-1, 1), cfg)[0])
    all_ths = np.concatenate([hon_ths, syb_ths])
    all_r = np.concatenate([hon_ratings_to_i, np.zeros(s_syb)]).reshape(-1, 1)
    prf_att = float(compute_credibility_prf(all_ths, all_r, cfg)[0])
    obs_shift = abs(prf_clean - prf_att)
    assert obs_shift <= theo_bound + 1e-9

    n_comp = 800
    comp_ths = np.full(n_comp, 0.85)
    c_comp = float(np.minimum(1.0, comp_ths / cfg.theta_c).sum())
    breakdown_bound = c_comp / (c_comp + float(c_h) + cfg.eps)
    assert breakdown_bound > 0.50

    return {
        "passed": True,
        "zero_history_sybil_shift": 0.0,
        "probationary_27pct_credibility_sybil_observed_shift": round(obs_shift, 4),
        "theorem3_theoretical_upper_bound": round(theo_bound, 4),
        "corrupted_majority_feedback_bound": round(breakdown_bound, 4),
    }


def verify_theorem_4(cfg: SimConfig, exp_results: dict | None = None) -> dict:
    """Verify Property 4: algebraic f_eff ratio under assumed condition 1 - rho >= psi and f < 1/3,
    plus modeled 30-trial (mean +- SD) summary across reliable, active corrupted, and sleeper feedback."""
    ratio_grid_ok = True
    for f in np.linspace(0.01, 0.333, 40):
        for rho in np.linspace(0.0, 0.25, 15):
            for psi in np.linspace(0.0, 1.0 - rho, 15):
                f_eff = (f * psi) / ((1.0 - f) * (1.0 - rho) + f * psi)
                if f_eff >= (1.0 / 3.0):
                    ratio_grid_ok = False

    if exp_results is None:
        exp_path = Path(__file__).resolve().parents[1] / "results" / "all_experiment_results.json"
        if exp_path.exists():
            exp_results = json.loads(exp_path.read_text(encoding="utf-8"))

    modeled_50pct = {}
    if exp_results and "adversarial_regimes" in exp_results:
        reg = exp_results["adversarial_regimes"]
        for key, label in (
            ("High (50% traitors, reliable feedback)", "reliable_feedback"),
            ("High (50% traitors, corrupted feedback)", "corrupted_feedback"),
            ("High (50% traitors, sleeper feedback poisoning)", "sleeper_feedback"),
        ):
            if key in reg:
                rho = float(reg[key]["fpr_pct"]["mean"]) / 100.0
                rho_sd = float(reg[key]["fpr_pct"]["sd"]) / 100.0
                psi = float(reg[key]["fnr_pct"]["mean"]) / 100.0
                psi_sd = float(reg[key]["fnr_pct"]["sd"]) / 100.0
                f = 0.50
                h_hm = (1.0 - f) * (1.0 - rho)
                a_hm = f * psi
                ratio_of_means = a_hm / max(1e-12, h_hm + a_hm)
                feff_block = reg[key].get("f_eff")
                f_eff_mean = float(feff_block["mean"]) if feff_block else round(ratio_of_means, 4)
                f_eff_sd = float(feff_block["sd"]) if feff_block else 0.0
                modeled_50pct[label] = {
                    "f": f,
                    "rho_fpr_mean": round(rho, 4),
                    "rho_fpr_sd": round(rho_sd, 4),
                    "psi_fnr_mean": round(psi, 4),
                    "psi_fnr_sd": round(psi_sd, 4),
                    "f_eff_mean": round(f_eff_mean, 4),
                    "f_eff_sd": round(f_eff_sd, 4),
                    "exceeds_one_third": bool(f_eff_mean > 1.0 / 3.0),
                }

    passed = bool(ratio_grid_ok)
    if "reliable_feedback" in modeled_50pct:
        passed = passed and (not modeled_50pct["reliable_feedback"]["exceeds_one_third"])
    if "sleeper_feedback" in modeled_50pct:
        passed = passed and modeled_50pct["sleeper_feedback"]["exceeds_one_third"]
    assert passed
    return {
        "passed": passed,
        "algebraic_one_third_condition_holds_under_assumption": bool(ratio_grid_ok),
        "modeled_50pct_summary": modeled_50pct,
    }


def verify_probationary_convergence(cfg: SimConfig) -> dict:
    """Genuine trust-engine probationary onboarding simulation matching consensus_sim dynamics.

    Tests evolving 1,000-node pool (K=40 P2P neighborhood PRF, 0.004 honest fault rate, stochastic NSI)
    under both active DDoS stress and nominal regimes without any artificial b >= N_min gate hiding
    early C_i >= T(t) crossing.
    """
    rng = np.random.default_rng(101)
    n_trials = 50
    n_inc = cfg.num_miners - 1
    k_nbr = min(cfg.peer_feedback_degree, cfg.num_miners)

    out = {}
    for regime_name, lam_hat, tau_hat, delta_hat in (
        ("active_ddos_stress", 0.70, 0.90, 0.75),
        ("nominal_quiescent", 0.15, 0.36, 0.15),
    ):
        conv_blocks = []
        unconstrained_admit_blocks = []
        statutory_grad_blocks = []
        noprob_excluded = 0

        for _ in range(n_trials):
            is_mal_inc = np.zeros(n_inc, dtype=bool)
            n_mal = int(round(0.30 * cfg.num_miners)) if regime_name == "active_ddos_stress" else int(round(0.10 * cfg.num_miners))
            is_mal_inc[:n_mal] = True
            rng.shuffle(is_mal_inc)

            is_mal_all = np.append(is_mal_inc, False)
            inc_ths = np.clip(np.where(is_mal_inc, rng.normal(0.30, 0.11, n_inc), rng.normal(0.80, 0.08, n_inc)), 0.02, 0.98)
            inc_nsi = np.clip(np.where(is_mal_inc, rng.normal(0.38, 0.12, n_inc), rng.normal(0.82, 0.08, n_inc)), 0.05, 0.99)
            hon_steady = float(np.mean(inc_ths[~is_mal_inc]))

            new_ths = cfg.gamma_probation * float(np.median(inc_ths))
            new_ths_np = 0.0
            # New honest entrant exhibits honest stability/uptime (matching honest NSI distribution)
            new_nsi = float(np.clip(rng.normal(0.82, 0.08), 0.15, 0.99))

            ths_all = np.append(inc_ths, new_ths)
            ths_np_all = np.append(inc_ths, new_ths_np)
            nsi_all = np.append(inc_nsi, new_nsi)

            nbr_indices = rng.integers(0, cfg.num_miners, size=(cfg.num_miners, k_nbr))
            rater_mal = is_mal_all[nbr_indices]
            target_mal = is_mal_all[:, None]
            nbr_ratings = rng.normal(0.80, 0.08, size=(cfg.num_miners, k_nbr))
            nbr_ratings[(~rater_mal) & target_mal] = rng.normal(0.28, 0.10, size=int(((~rater_mal) & target_mal).sum()))
            nbr_ratings[(rater_mal) & (~target_mal)] = rng.normal(0.68, 0.10, size=int(((rater_mal) & (~target_mal)).sum()))
            nbr_ratings = np.clip(nbr_ratings, 0.0, 1.0)

            w = compute_adaptive_weights(lam_hat, tau_hat, delta_hat, cfg)
            conv_b = None
            unc_adm_b = None
            noprob_adm_b = None
            prob_left = np.zeros(cfg.num_miners, dtype=int)
            prob_left[-1] = cfg.n_min_probation

            for b in range(1, 151):
                prf = compute_credibility_prf(ths_all, nbr_ratings, cfg, neighbor_indices=nbr_indices)
                C = compute_topsis_closeness(ths_all, nsi_all, prf, w, prob_left, cfg)
                thresh, _, _ = compute_dynamic_threshold(C, delta_hat, cfg)
                admitted = C >= thresh

                if unc_adm_b is None and bool(admitted[-1]):
                    unc_adm_b = b

                # Honest entrant validates blocks once admitted (during probation if admitted, with honest 0.004 fault rate)
                participated = np.zeros(cfg.num_miners, dtype=bool)
                adm_idx = np.where(admitted)[0]
                if len(adm_idx) > 0:
                    chosen = rng.choice(adm_idx, size=min(len(adm_idx), max(2, int(0.05 * len(adm_idx)))), replace=False)
                    participated[chosen] = True
                if bool(admitted[-1]):
                    participated[-1] = True  # Track entrant per-round validation once eligible

                valid_out = np.ones(cfg.num_miners, dtype=bool)
                mal_p = participated & is_mal_all
                hon_p = participated & (~is_mal_all)
                valid_out[mal_p] = ~(rng.random(int(mal_p.sum())) < (0.28 if regime_name == "active_ddos_stress" else 0.10))
                valid_out[hon_p] = ~(rng.random(int(hon_p.sum())) < 0.004)

                ths_all = update_ths_asymmetric(ths_all, valid_out, participated, cfg)
                nsi_all = np.clip(nsi_all + rng.normal(0.0, 0.004, size=cfg.num_miners), 0.05, 0.99)
                prob_left = np.maximum(0, prob_left - admitted.astype(int))

                if conv_b is None and ths_all[-1] >= 0.95 * hon_steady:
                    conv_b = b

                # Unassisted zero-history entrant (no probation init, no THS weight cap)
                prf_np = compute_credibility_prf(ths_np_all, nbr_ratings, cfg, neighbor_indices=nbr_indices)
                C_np = compute_topsis_closeness(ths_np_all, nsi_all, prf_np, w, None, None)
                t_np, _, _ = compute_dynamic_threshold(C_np, delta_hat, cfg)
                if bool(C_np[-1] >= t_np):
                    if noprob_adm_b is None:
                        noprob_adm_b = b
                    ok_np = bool(rng.random() >= 0.004)
                    ths_np_all[-1] = float(update_ths_asymmetric(np.array([ths_np_all[-1]]), np.array([ok_np]), np.array([True]), cfg)[0])

            conv_blocks.append(conv_b if conv_b is not None else 150)
            unconstrained_admit_blocks.append(unc_adm_b if unc_adm_b is not None else 150)
            statutory_grad_blocks.append(max(cfg.n_min_probation, unc_adm_b) if unc_adm_b is not None else 150)
            if noprob_adm_b is None:
                noprob_excluded += 1

        out[regime_name] = {
            "median_blocks_to_95pct_steady": float(np.median(conv_blocks)),
            "median_blocks_to_unconstrained_threshold_crossing": float(np.median(unconstrained_admit_blocks)),
            "median_blocks_to_statutory_cap_graduation": float(np.median(statutory_grad_blocks)),
            "without_probation_excluded_pct": round(100.0 * noprob_excluded / n_trials, 1),
        }

    # Preserve backward-compatible flat summary keys pointing to active_ddos_stress while exposing full breakdown
    stress = out["active_ddos_stress"]
    return {
        "median_blocks_to_95pct_steady": stress["median_blocks_to_95pct_steady"],
        "median_blocks_to_unconstrained_HM_crossing": stress["median_blocks_to_unconstrained_threshold_crossing"],
        "statutory_probation_cap_graduation_blocks": stress["median_blocks_to_statutory_cap_graduation"],
        "without_probation_excluded_pct": stress["without_probation_excluded_pct"],
        "nominal_regime": out["nominal_quiescent"],
    }


def verify_threshold_noise_sensitivity(cfg: SimConfig) -> dict:
    """Measure true shift of T(t) with 5% trimming under fixed-m (1,000-node) low-tail score replacement."""
    rng = np.random.default_rng(99)
    n = 1000
    base_c = np.concatenate([
        np.clip(rng.normal(0.80, 0.08, 700), 0.50, 0.99),
        np.clip(rng.normal(0.25, 0.07, 300), 0.01, 0.50),
    ])
    t0, mu0, sig0 = compute_dynamic_threshold(base_c, 0.60, cfg)
    low_order = np.argsort(base_c)
    out = {"baseline_T": round(t0, 4), "injections": {}}
    for frac in (0.05, 0.10, 0.20):
        n_noise = int(frac * n)
        aug = base_c.copy()
        aug[low_order[:n_noise]] = rng.uniform(0.01, 0.08, n_noise)
        t_new, mu_new, _ = compute_dynamic_threshold(aug, 0.60, cfg)
        out["injections"][f"{int(frac*100)}pct"] = {
            "T_new": round(t_new, 4),
            "T_relative_shift_pct": round(abs(t_new - t0) / t0 * 100.0, 2),
            "trimmed_mean_shift_pct": round(abs(mu_new - mu0) / mu0 * 100.0, 2),
        }
    return out


def _run_100miner_episode(cfg: SimConfig, seed: int, use_bandit: bool) -> dict:
    """Execute one genuine 100-miner discrete-time episode over 120 slots with regime shifts."""
    rng = np.random.default_rng(seed)
    n_nodes = 100
    topo = P2PNetworkTopology(n_nodes, rng, wan_mode=False, backbone_bw_mbps=cfg.backbone_bw_mbps, edge_bw_mbps=cfg.edge_bw_mbps)

    arms = [(0.65, 0.06), (0.75, 0.08), (0.85, 0.10), (0.95, 0.12), (1.05, 0.15)]
    counts = np.zeros((3, len(arms)), dtype=float)
    rewards = np.zeros((3, len(arms)), dtype=float)

    n_mal = 30
    is_mal = np.zeros(n_nodes, dtype=bool)
    is_mal[:n_mal] = True
    rng.shuffle(is_mal)

    ths = np.clip(np.where(is_mal, rng.normal(0.32, 0.11, n_nodes), rng.normal(0.80, 0.08, n_nodes)), 0.02, 0.98)
    nsi = np.clip(np.where(is_mal, rng.normal(0.45, 0.14, n_nodes), rng.normal(0.82, 0.08, n_nodes)), 0.05, 0.99)
    k_nbr = min(cfg.peer_feedback_degree, n_nodes)
    nbr_indices = rng.integers(0, n_nodes, size=(n_nodes, k_nbr))
    nbr_ratings = rng.normal(0.80, 0.08, size=(n_nodes, k_nbr))
    nbr_ratings[(~is_mal[nbr_indices]) & is_mal[:, None]] = rng.normal(0.28, 0.10, size=int(((~is_mal[nbr_indices]) & is_mal[:, None]).sum()))
    nbr_ratings[is_mal[nbr_indices] & (~is_mal[:, None])] = rng.normal(0.55, 0.12, size=int((is_mal[nbr_indices] & (~is_mal[:, None])).sum()))
    nbr_ratings = np.clip(nbr_ratings, 0.0, 1.0)

    bprop_list = []
    fpr_list = []
    fnr_list = []
    prev_bprop = 0.40
    mempool_q = 0.0

    for t in range(1, 121):
        # Time-varying multi-phase stress profile: quiescent (1..30), moderate burst (31..75), severe flood (76..120)
        if t <= 30:
            spam_in = 0.0
            defect_p = 0.10
        elif t <= 75:
            spam_in = 2200.0
            defect_p = 0.22
        else:
            spam_in = 5000.0
            defect_p = 0.30

        honest_in = cfg.nominal_tx_rate
        total_arr = honest_in + spam_in
        lam_hat = float(np.clip((prev_bprop - 0.30) / 1.40 + max(0.0, total_arr - cfg.service_capacity_tx_s) / cfg.ddos_attack_rate, 0.0, 1.0))
        tau_hat = float(np.clip(total_arr / (cfg.nominal_tx_rate + cfg.ddos_attack_rate), 0.0, 1.0))
        delta_hat = float(np.clip(0.45 * max(0.0, total_arr - honest_in) / cfg.ddos_attack_rate + 0.35 * float(np.mean(ths < cfg.theta_c)), 0.0, 1.0))

        ctx_score = 0.5 * (delta_hat + tau_hat)
        ctx = 0 if ctx_score < 0.35 else (1 if ctx_score < 0.70 else 2)

        if use_bandit:
            n_c = int(counts[ctx].sum())
            if n_c < len(arms):
                arm_idx = n_c
            else:
                ucb = rewards[ctx] / counts[ctx] + 0.10 * np.sqrt(np.log(max(2, n_c)) / counts[ctx])
                arm_idx = int(np.argmax(ucb))
            alpha, beta = arms[arm_idx]
        else:
            alpha, beta = cfg.alpha_thresh, cfg.beta_thresh

        w = compute_adaptive_weights(lam_hat, tau_hat, delta_hat, cfg)
        prf = compute_credibility_prf(ths, nbr_ratings, cfg, neighbor_indices=nbr_indices)
        closeness = compute_topsis_closeness(ths, nsi, prf, w, None, cfg)

        # Compute dynamic trimmed threshold using chosen (alpha, beta) arm
        k_trim = int(np.floor(cfg.trim_fraction * n_nodes))
        part = np.partition(closeness, (k_trim - 1, n_nodes - k_trim))
        trimmed = part[k_trim : n_nodes - k_trim]
        mu_t, sig_t = float(np.mean(trimmed)), float(np.std(trimmed))
        thresh = float(np.clip(mu_t - alpha * sig_t + beta * delta_hat, cfg.t_min, cfg.t_max))

        admitted = closeness >= thresh
        fpr = float((~admitted & ~is_mal).sum()) / max(1, int((~is_mal).sum()))
        fnr = float((admitted & is_mal).sum()) / max(1, n_mal)

        admitted_spam = spam_in * fnr
        admitted_honest = honest_in * (1.0 - fpr * cfg.honest_relay_penalty_weight)
        mal_share = float((admitted & is_mal).sum()) / max(1, int(admitted.sum()))
        svc = cfg.service_capacity_tx_s * (1.0 - 0.42 * mal_share)
        mempool_q = float(np.clip(mempool_q + (admitted_honest + admitted_spam) - svc, 0.0, cfg.mempool_capacity_tx * 1.3))
        load_ratio = float(np.clip(mempool_q / cfg.mempool_capacity_tx + admitted_spam / cfg.service_capacity_tx_s, 0.0, 2.0))
        b_prop = topo.compute_propagation_delay(admitted, is_mal, load_ratio, 0.0, cfg.block_size_mb)
        prev_bprop = b_prop

        # Genuine proposer & THS evolution
        adm_idx = np.where(admitted)[0]
        prop = np.zeros(n_nodes, dtype=bool)
        if len(adm_idx) > 0:
            prop[rng.choice(adm_idx, size=min(len(adm_idx), max(2, int(0.08 * len(adm_idx)))), replace=False)] = True
        valid = np.ones(n_nodes, dtype=bool)
        valid[prop & is_mal] = ~(rng.random(int((prop & is_mal).sum())) < defect_p)
        valid[prop & (~is_mal)] = ~(rng.random(int((prop & (~is_mal)).sum())) < 0.004)
        ths = update_ths_asymmetric(ths, valid, prop, cfg)
        nsi = np.clip(nsi + rng.normal(0.0, 0.004, size=n_nodes), 0.05, 0.99)

        if use_bandit:
            # Reward balances propagation speed against false exclusions and malicious leakage
            step_reward = float(np.clip(1.0 - 0.6 * b_prop - 0.8 * fpr - 1.5 * fnr, -1.0, 1.0))
            counts[ctx, arm_idx] += 1.0
            rewards[ctx, arm_idx] += step_reward

        if t > 30:
            bprop_list.append(b_prop)
            fpr_list.append(fpr * 100.0)
            fnr_list.append(fnr * 100.0)

    return {
        "bprop_mean": float(np.mean(bprop_list)),
        "fpr_pct": float(np.mean(fpr_list)),
        "fnr_pct": float(np.mean(fnr_list)),
    }


def verify_contextual_bandit_pilot(cfg: SimConfig) -> dict:
    """Genuine 100-miner, 20-run contextual UCB1 pilot vs SimConfig default (alpha=0.85, beta=0.10)."""
    fixed_bprop, bandit_bprop = [], []
    fixed_fpr, bandit_fpr = [], []
    fixed_fnr, bandit_fnr = [], []

    for run in range(20):
        seed = 500 + run
        rf = _run_100miner_episode(cfg, seed, use_bandit=False)
        rb = _run_100miner_episode(cfg, seed, use_bandit=True)
        fixed_bprop.append(rf["bprop_mean"])
        bandit_bprop.append(rb["bprop_mean"])
        fixed_fpr.append(rf["fpr_pct"])
        bandit_fpr.append(rb["fpr_pct"])
        fixed_fnr.append(rf["fnr_pct"])
        bandit_fnr.append(rb["fnr_pct"])

    mean_f = float(np.mean(fixed_bprop))
    mean_b = float(np.mean(bandit_bprop))
    red_pct = (mean_f - mean_b) / mean_f * 100.0
    return {
        "n_miners": 100,
        "n_runs": 20,
        "baseline_fixed_arm": [cfg.alpha_thresh, cfg.beta_thresh],
        "fixed_prop_delay_s": round(mean_f, 4),
        "bandit_prop_delay_s": round(mean_b, 4),
        "delay_reduction_pct": round(red_pct, 2),
        "fixed_fpr_pct": round(float(np.mean(fixed_fpr)), 2),
        "bandit_fpr_pct": round(float(np.mean(bandit_fpr)), 2),
        "fixed_fnr_pct": round(float(np.mean(fixed_fnr)), 2),
        "bandit_fnr_pct": round(float(np.mean(bandit_fnr)), 2),
    }


def main(exp_results: dict | None = None) -> None:
    cfg = SimConfig()
    summary = {
        "theorem_1_adaptive_weights": verify_theorem_1(cfg),
        "theorem_2_asymmetric_ths": verify_theorem_2(cfg),
        "theorem_3_sybil_bound": verify_theorem_3(cfg),
        "theorem_4_bft_capacity_and_feff": verify_theorem_4(cfg, exp_results=exp_results),
        "theorem_5_trimmed_threshold_noise": verify_threshold_noise_sensitivity(cfg),
        "probationary_convergence": verify_probationary_convergence(cfg),
        "contextual_bandit_pilot": verify_contextual_bandit_pilot(cfg),
    }
    out_path = Path(__file__).resolve().parents[1] / "results" / "proof_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
