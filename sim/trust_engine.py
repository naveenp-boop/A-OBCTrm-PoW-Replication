"""Core mathematical engine for A-OBCTrm-PoW:
- Adaptive TOPSIS weighting (Eqs. 1-5)
- Credibility-weighted Peer Reliability Factor PRF (Eqs. 6-8)
- Asymmetric trust penalty dynamics (Eq. 9)
- Probationary entry mechanism (Eqs. 10-11)
- Trimmed dynamic admission threshold T(t) (Eq. 12)
"""

from __future__ import annotations
import numpy as np
from .config import SimConfig


def compute_adaptive_weights(
    lam_hat: float,
    tau_hat: float,
    delta_hat: float,
    cfg: SimConfig,
) -> np.ndarray:
    """Compute normalized adaptive weights [w_THS, w_NSI, w_PRF] with simplex upper cap."""
    lam_hat = float(np.clip(lam_hat, 0.0, 1.0))
    tau_hat = float(np.clip(tau_hat, 0.0, 1.0))
    delta_hat = float(np.clip(delta_hat, 0.0, 1.0))

    f_ths = float(np.exp(cfg.k1_ths * delta_hat))
    f_nsi = 1.0 + cfg.k2_nsi * lam_hat
    f_prf = 1.0 + cfg.k3_prf * tau_hat

    f_vec = np.array([f_ths, f_nsi, f_prf], dtype=np.float64)
    w = f_vec / f_vec.sum()

    # Enforce hard criterion share cap (w_max = 0.80) to avoid single-criterion collapse
    if w[0] > cfg.w_max_cap:
        excess = w[0] - cfg.w_max_cap
        w[0] = cfg.w_max_cap
        other_sum = w[1] + w[2]
        if other_sum > 0:
            w[1] += excess * (w[1] / other_sum)
            w[2] += excess * (w[2] / other_sum)
        else:
            w[1] = w[2] = 0.5 * (1.0 - cfg.w_max_cap)

    return w


def compute_credibility_prf(
    ths: np.ndarray,
    raw_peer_ratings: np.ndarray,
    cfg: SimConfig,
    uniform_fallback: bool = False,
    neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate peer ratings into PRF_i weighted by rater credibility c_k = min(1, THS_k / theta_c).

    Supports both dense (m x m) rating matrices and P2P neighborhood matrices (m x K)
    indexed via neighbor_indices[i, k] = rater id for target i's k-th peer.
    """
    if neighbor_indices is None:
        if uniform_fallback:
            return np.clip(raw_peer_ratings.mean(axis=0), 0.0, 1.0)
        c_k = np.minimum(1.0, np.maximum(0.0, ths) / cfg.theta_c)
        weighted_sum = c_k @ raw_peer_ratings
        cred_mass = float(c_k.sum()) + cfg.eps
        return np.clip(weighted_sum / cred_mass, 0.0, 1.0)

    # Sparse neighborhood N(i) aggregation: raw_peer_ratings has shape (m, K) -> raters for node i
    if uniform_fallback:
        return np.clip(raw_peer_ratings.mean(axis=1), 0.0, 1.0)
    c_nbr = np.minimum(1.0, np.maximum(0.0, ths[neighbor_indices]) / cfg.theta_c)
    weighted_sum = np.sum(c_nbr * raw_peer_ratings, axis=1)
    cred_mass = np.sum(c_nbr, axis=1) + cfg.eps
    return np.clip(weighted_sum / cred_mass, 0.0, 1.0)


def update_ths_asymmetric(
    ths: np.ndarray,
    valid_outcomes: np.ndarray,
    participated: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Update Historical Judgment THS_i(t+1) asynchronously with multiplicative collapse penalty."""
    new_ths = ths.copy()
    valid_mask = participated & valid_outcomes
    fail_mask = participated & (~valid_outcomes)

    new_ths[valid_mask] = np.minimum(
        1.0, ths[valid_mask] + cfg.eta_reward * (1.0 - ths[valid_mask])
    )
    new_ths[fail_mask] = np.maximum(
        0.0, cfg.kappa_penalty * ths[fail_mask] - cfg.zeta_penalty
    )
    return new_ths


def compute_topsis_closeness(
    ths: np.ndarray,
    nsi: np.ndarray,
    prf: np.ndarray,
    weights: np.ndarray,
    probation_blocks_left: np.ndarray | None = None,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Compute TOPSIS Closeness Coefficient C_i in [0, 1] for every miner."""
    D = np.column_stack([ths, nsi, prf]).astype(np.float64)
    col_norms = np.linalg.norm(D, axis=0) + 1e-12
    R = D / col_norms

    # Apply per-node weight matrix (honoring probationary THS cap when active)
    W_mat = np.broadcast_to(weights, D.shape).copy()
    if probation_blocks_left is not None and cfg is not None:
        prob_mask = probation_blocks_left > 0
        if np.any(prob_mask):
            capped = np.minimum(W_mat[prob_mask, 0], cfg.w_ths_probation_cap)
            redistr = W_mat[prob_mask, 0] - capped
            W_mat[prob_mask, 0] = capped
            other = W_mat[prob_mask, 1] + W_mat[prob_mask, 2] + 1e-12
            W_mat[prob_mask, 1] += redistr * (W_mat[prob_mask, 1] / other)
            W_mat[prob_mask, 2] += redistr * (W_mat[prob_mask, 2] / other)

    V = R * W_mat
    A_pos = V.max(axis=0)
    A_neg = V.min(axis=0)

    S_pos = np.linalg.norm(V - A_pos, axis=1)
    S_neg = np.linalg.norm(V - A_neg, axis=1)

    return np.clip(S_neg / (S_pos + S_neg + 1e-12), 0.0, 1.0)


def compute_dynamic_threshold(
    closeness: np.ndarray,
    delta_hat: float,
    cfg: SimConfig,
) -> tuple[float, float, float]:
    """Compute dynamic threshold T(t) using symmetric 5%-trimmed distribution statistics."""
    xs = np.sort(closeness)
    k = int(np.floor(cfg.trim_fraction * len(xs)))
    trimmed = xs[k : len(xs) - k] if k > 0 and 2 * k < len(xs) else xs

    mu_trim = float(np.mean(trimmed))
    sig_trim = float(np.std(trimmed))
    raw_t = mu_trim - cfg.alpha_thresh * sig_trim + cfg.beta_thresh * float(delta_hat)
    threshold = float(np.clip(raw_t, cfg.t_min, cfg.t_max))
    return threshold, mu_trim, sig_trim
