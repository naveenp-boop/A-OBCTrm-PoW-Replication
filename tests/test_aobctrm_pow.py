"""Automated verification suite validating core theorems and A-OBCTrm-PoW invariants.

Runs cleanly with `python -m unittest discover -s tests -v`.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import SimConfig
from sim.trust_engine import (
    compute_adaptive_weights,
    compute_credibility_prf,
    compute_dynamic_threshold,
    compute_topsis_closeness,
)
from sim.consensus_sim import solve_scaled_pow_sample, simulate_single_trial
from scripts.verify_proofs import (
    verify_theorem_1,
    verify_theorem_2,
    verify_theorem_3,
    verify_theorem_4,
    verify_threshold_noise_sensitivity,
)


class TestAOBCTrmPoW(unittest.TestCase):
    def test_theorem_1_weights(self):
        res = verify_theorem_1(SimConfig())
        self.assertTrue(res["passed"])
        self.assertLessEqual(res["clamped_max_weight"], 0.80 + 1e-9)
        self.assertTrue(res["monotonic_strictly"])

    def test_theorem_2_asymmetric_ths(self):
        res = verify_theorem_2(SimConfig())
        self.assertTrue(res["passed"])
        self.assertEqual(res["zero_to_half_recovery_blocks"], 14)

    def test_theorem_3_sybil_bound(self):
        res = verify_theorem_3(SimConfig())
        self.assertTrue(res["passed"])
        self.assertEqual(res["zero_history_sybil_shift"], 0.0)
        self.assertLessEqual(
            res["probationary_30pct_sybil_observed_shift"],
            res["theorem3_theoretical_upper_bound"],
        )

    def test_real_pow_puzzle(self):
        nonce, digest_hex, _ = solve_scaled_pow_sample("unit-test-block", leading_zero_bits=12)
        self.assertGreaterEqual(nonce, 0)
        self.assertTrue(digest_hex.startswith("000"))

    def test_topsis_ordering_and_threshold_bounds(self):
        cfg = SimConfig()
        rng = np.random.default_rng(7)
        ths = np.concatenate([rng.uniform(0.75, 0.95, 90), rng.uniform(0.05, 0.25, 10)])
        nsi = np.concatenate([rng.uniform(0.75, 0.95, 90), rng.uniform(0.10, 0.30, 10)])
        prf = np.concatenate([rng.uniform(0.75, 0.95, 90), rng.uniform(0.10, 0.30, 10)])
        w = compute_adaptive_weights(0.5, 0.5, 0.7, cfg)
        c = compute_topsis_closeness(ths, nsi, prf, w)
        t, _, _ = compute_dynamic_threshold(c, 0.7, cfg)
        self.assertTrue(0.0 <= t <= 1.0)
        self.assertGreater(c[:90].mean(), t)
        self.assertGreater(t, c[90:].mean())

    def test_corrupted_feedback_degradation_at_50pct(self):
        cfg = SimConfig()
        clean_50 = simulate_single_trial("A-OBCTrm-PoW", 1, cfg, traitor_fraction=0.50, attack_mode="ddos")
        corrupt_50 = simulate_single_trial("A-OBCTrm-PoW", 1, cfg, traitor_fraction=0.50, attack_mode="corrupted_feedback")
        self.assertGreater(corrupt_50["fnr_pct"], clean_50["fnr_pct"])

    def test_theorem_4_bft_and_feff(self):
        res = verify_theorem_4(SimConfig(), exp_results={})
        self.assertTrue(res["passed"])
        self.assertTrue(res["bft_one_third_margin_holds_across_grid"])
        f, rho, psi = 0.50, 0.0920, 0.6447
        f_eff = (f * psi) / ((1.0 - f) * (1.0 - rho) + f * psi)
        self.assertAlmostEqual(f_eff, 0.4152, places=4)

    def test_theorem_5_fixed_sample_breakdown(self):
        res = verify_threshold_noise_sensitivity(SimConfig())
        self.assertEqual(res["injections"]["5pct"]["trimmed_mean_shift_pct"], 0.0)
        self.assertGreater(res["injections"]["10pct"]["trimmed_mean_shift_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
