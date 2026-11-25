"""Centralized configuration and calibrated hyperparameters for A-OBCTrm-PoW."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Network / SimBlock-equivalent parameters
    num_miners: int = 1000
    num_trials: int = 30
    blocks_per_trial: int = 500
    sim_duration_s: int = 120
    ddos_start_s: int = 30

    # Scaled PoW target (1/60 of 600s Bitcoin target)
    target_block_interval_s: float = 10.0
    pow_leading_zero_bits: int = 16

    # Workload & DDoS injection rates
    nominal_tx_rate: float = 2850.0        # baseline transaction arrival rate (tx/s)
    service_capacity_tx_s: float = 3400.0  # nominal validation clearing capacity (tx/s)
    ddos_attack_rate: float = 5000.0       # malicious mempool flood rate (tx/s)
    backbone_bw_mbps: float = 100.0
    edge_bw_mbps: float = 15.0
    mempool_capacity_tx: int = 15000
    tx_size_bytes: int = 250
    block_size_mb: float = 1.25
    peer_feedback_degree: int = 40         # |N(i)| P2P feedback neighborhood size

    # Adaptive TOPSIS weights (Section 3.3)
    k1_ths: float = 2.5
    k2_nsi: float = 1.5
    k3_prf: float = 1.0
    w_max_cap: float = 0.80
    window_w: int = 50

    # Credibility-weighted PRF & asymmetric trust update (Section 3.3.2-3.3.3)
    theta_c: float = 0.55
    eta_reward: float = 0.05
    kappa_penalty: float = 0.40
    zeta_penalty: float = 0.05
    eps: float = 1e-5

    # Dynamic trimmed threshold (Section 3.3.4)
    # T(t) = clip(mu_trim - alpha * sigma_trim + beta * delta_hat, t_min, t_max)
    alpha_thresh: float = 0.85
    beta_thresh: float = 0.10
    trim_fraction: float = 0.05
    t_min: float = 0.47
    t_max: float = 0.56

    # Probationary entry (Section 3.4)
    gamma_probation: float = 0.50
    n_min_probation: int = 50
    w_ths_probation_cap: float = 0.30

    # Baselines (Section 4.2)
    ewma_lambda: float = 0.85
    token_bucket_capacity: int = 500
    token_bucket_rate: float = 50.0
    dyn_fee_zeta: float = 2.0

    # 50-node WAN emulation parameters (Section 4.8 & 5.1)
    wan_nodes: int = 50
    wan_mu_ln: float = 4.5
    wan_sigma_ln: float = 0.9
    wan_packet_loss: float = 0.005
