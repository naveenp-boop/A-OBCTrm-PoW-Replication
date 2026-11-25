"""Baseline defense and consensus filtering implementations:
1. Standard PoW (Undefended FCFS)
2. ML-IDS (100-tree Random Forest on 14 mempool & traffic features)
3. Static TOPSIS (Fixed weights [1/3, 1/3, 1/3], unweighted PRF, no probation)
4. EWMA Reputation Filter (lambda=0.85)
5. Token-Bucket Rate Limiting (B=500, r=50 tx/s)
6. Dynamic Fee Mechanism (zeta=2.0)
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .config import SimConfig
from .trust_engine import compute_topsis_closeness


class MLIDSDetector:
    """Real scikit-learn 100-tree Random Forest classifier trained on 14 mempool/flow features."""

    FEATURE_NAMES = [
        "pkt_rate",
        "inter_arrival_mean",
        "inter_arrival_std",
        "byte_rate",
        "tx_fee_ratio",
        "tx_size_var",
        "syn_ratio",
        "burst_duration",
        "dup_nonce_rate",
        "hop_count",
        "peer_entropy",
        "mempool_occ",
        "orphan_rate",
        "jitter_ms",
    ]

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        n_train = 2400
        y = np.concatenate([np.zeros(1400, dtype=int), np.ones(1000, dtype=int)])
        X = np.zeros((n_train, 14), dtype=np.float64)

        # Honest flow distributions (14 features)
        hon = y == 0
        nh = int(hon.sum())
        X[hon] = np.column_stack([
            rng.normal(25.0, 6.0, nh),      # pkt_rate
            rng.normal(40.0, 9.0, nh),      # inter_arrival_mean
            rng.normal(8.0, 2.5, nh),       # inter_arrival_std
            rng.normal(6250.0, 1400.0, nh), # byte_rate
            rng.normal(1.05, 0.20, nh),     # tx_fee_ratio
            rng.normal(0.18, 0.05, nh),     # tx_size_var
            rng.normal(0.12, 0.04, nh),     # syn_ratio
            rng.normal(1.5, 0.5, nh),       # burst_duration
            rng.normal(0.01, 0.008, nh),    # dup_nonce_rate
            rng.normal(3.5, 0.7, nh),       # hop_count
            rng.normal(3.1, 0.35, nh),      # peer_entropy
            rng.normal(0.32, 0.12, nh),     # mempool_occ
            rng.normal(0.015, 0.007, nh),   # orphan_rate
            rng.normal(9.5, 3.0, nh),       # jitter_ms
        ])

        # Attack/DDoS/spam flow distributions (realistic partial overlap)
        att = y == 1
        na = int(att.sum())
        X[att] = np.column_stack([
            rng.normal(135.0, 42.0, na),
            rng.normal(8.5, 4.0, na),
            rng.normal(2.2, 1.3, na),
            rng.normal(29000.0, 9500.0, na),
            rng.normal(0.45, 0.25, na),
            rng.normal(0.06, 0.04, na),
            rng.normal(0.68, 0.16, na),
            rng.normal(7.8, 2.4, na),
            rng.normal(0.19, 0.08, na),
            rng.normal(5.1, 1.1, na),
            rng.normal(1.5, 0.55, na),
            rng.normal(0.84, 0.14, na),
            rng.normal(0.11, 0.045, na),
            rng.normal(38.0, 14.0, na),
        ])

        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=1,
            random_state=seed,
        )
        self.clf.fit(X, y)

    def extract_and_classify(
        self,
        num_nodes: int,
        is_malicious: np.ndarray,
        attack_active: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Extract 14 traffic features per node and run 100-tree RF inference."""
        X = np.zeros((num_nodes, 14), dtype=np.float64)
        mal = is_malicious.copy()
        if not attack_active:
            # Pre-attack: malicious actors mimic lower-rate traffic (harder to detect early)
            effective_att = mal & (rng.random(num_nodes) < 0.20)
        else:
            effective_att = mal

        nh = int((~effective_att).sum())
        na = int(effective_att.sum())
        if nh > 0:
            X[~effective_att] = np.column_stack([
                rng.normal(26.0, 7.0, nh),
                rng.normal(39.0, 10.0, nh),
                rng.normal(8.2, 2.8, nh),
                rng.normal(6400.0, 1500.0, nh),
                rng.normal(1.03, 0.22, nh),
                rng.normal(0.18, 0.06, nh),
                rng.normal(0.13, 0.05, nh),
                rng.normal(1.6, 0.6, nh),
                rng.normal(0.012, 0.01, nh),
                rng.normal(3.6, 0.8, nh),
                rng.normal(3.05, 0.4, nh),
                rng.normal(0.38 if attack_active else 0.30, 0.14, nh),
                rng.normal(0.018, 0.009, nh),
                rng.normal(11.0 if attack_active else 9.0, 3.5, nh),
            ])
        if na > 0:
            X[effective_att] = np.column_stack([
                rng.normal(128.0, 46.0, na),
                rng.normal(10.0, 5.0, na),
                rng.normal(2.5, 1.5, na),
                rng.normal(27500.0, 9800.0, na),
                rng.normal(0.48, 0.26, na),
                rng.normal(0.07, 0.04, na),
                rng.normal(0.64, 0.18, na),
                rng.normal(7.2, 2.6, na),
                rng.normal(0.18, 0.09, na),
                rng.normal(5.0, 1.1, na),
                rng.normal(1.6, 0.60, na),
                rng.normal(0.81, 0.16, na),
                rng.normal(0.10, 0.05, na),
                rng.normal(36.0, 15.0, na),
            ])

        preds = self.clf.predict(X)
        # Admitted = predicted benign (0)
        return preds == 0


def static_topsis_admit(
    ths: np.ndarray,
    nsi: np.ndarray,
    raw_prf: np.ndarray,
    threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Static TOPSIS baseline with fixed equal weights [1/3, 1/3, 1/3] and uniform PRF."""
    w_static = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    C = compute_topsis_closeness(ths, nsi, raw_prf, w_static, probation_blocks_left=None, cfg=None)
    return C >= threshold, C


def ewma_update_and_admit(
    ewma_scores: np.ndarray,
    round_quality: np.ndarray,
    cfg: SimConfig,
    threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-metric EWMA filter: R_i(t) = lam*R_i(t-1) + (1-lam)*m_i(t)."""
    updated = cfg.ewma_lambda * ewma_scores + (1.0 - cfg.ewma_lambda) * round_quality
    return updated >= threshold, updated


def token_bucket_filter(
    honest_tx_rate: float,
    attack_tx_rate: float,
    capacity_tx: int,
    refill_rate_tx_s: float,
    num_ingress_peers: int = 55,
) -> tuple[float, float, float]:
    """Indiscriminate token-bucket ingress throttling aggregated across gateway peers.

    Returns (admitted_honest_tx_s, admitted_attack_tx_s, honest_penalty_frac).
    """
    # Per-gateway refill budget + transient burst buffer credit
    total_refill = num_ingress_peers * refill_rate_tx_s + 0.08 * num_ingress_peers * (capacity_tx / 10.0)
    total_demand = honest_tx_rate + attack_tx_rate
    if total_demand <= total_refill:
        return honest_tx_rate, attack_tx_rate, 0.0
    accept_ratio = total_refill / total_demand
    admitted_h = honest_tx_rate * accept_ratio
    admitted_a = attack_tx_rate * accept_ratio
    penalty = max(0.0, (honest_tx_rate - admitted_h) / max(1.0, honest_tx_rate))
    return admitted_h, admitted_a, penalty


def dynamic_fee_filter(
    honest_tx_rate: float,
    attack_tx_rate: float,
    queue_occ: float,
    cfg: SimConfig,
) -> tuple[float, float, float]:
    """Congestion-based quadratic fee pricing Fee(t) = Fee_base * [1 + zeta*(Q/Qmax)^2].

    Returns (admitted_honest_tx_s, admitted_attack_tx_s, honest_penalty_frac).
    """
    fee_mult = 1.0 + cfg.dyn_fee_zeta * (np.clip(queue_occ, 0.0, 1.5) ** 2)
    # Regular honest users have moderate fee elasticity; spam drops slightly faster under fee spikes
    honest_pass = float(np.clip(1.0 / (1.0 + 0.045 * max(0.0, fee_mult - 1.0)), 0.75, 1.0))
    attack_pass = float(np.clip(1.0 / (1.0 + 0.30 * max(0.0, fee_mult - 1.0)), 0.45, 1.0))

    admitted_h = honest_tx_rate * honest_pass
    admitted_a = attack_tx_rate * attack_pass
    return admitted_h, admitted_a, 1.0 - honest_pass
