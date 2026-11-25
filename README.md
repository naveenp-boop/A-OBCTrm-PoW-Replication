# A-OBCTrm-PoW: Replication Package, Simulation Suite & Formal Proofs

This repository provides the full reproducible codebase, analytical property/bound derivations (`PROOFS.md`), automated verification tests, pinned Python 3.14.3 dependencies, optional unexecuted Linux topology templates, and simulated telemetry for the manuscript:

> **Lightweight Trust-Aware Consensus Optimization for Efficient Blockchain Performance Under Adversarial Conditions**  
> Puppala Naga Sravanthi and Pasam Naveen  
> Department of Computer Science and Engineering, School of Engineering and Applied Sciences, SRM University-AP, Amaravati, 522240, India

---

## Repository Structure

```text
A-OBCTrm-PoW-Replication/
├── PROOFS.md                         # Analytical properties, algebraic bounds & derivations (Properties 1–5)
├── CITATION.cff                      # Software citation metadata
├── LICENSE                           # MIT License
├── requirements.txt                  # Minimum Python package dependencies
├── requirements-lock.txt             # Pinned dependency lockfile (Python 3.14.3)
├── sim/
│   ├── config.py                     # All consensus, TOPSIS, threshold & topology hyperparameters
│   ├── topology.py                   # 1,000-node SimBlock-calibrated overlay + 50-node WAN graph
│   ├── trust_engine.py               # Dynamic TOPSIS, Credibility PRF, asymmetric THS & probation
│   ├── baselines.py                  # Standard PoW, 100-tree RF ML-IDS, Static TOPSIS, EWMA, Token Bucket, Dynamic Fee
│   ├── consensus_sim.py              # Discrete-time analytic consensus & mempool model + SHA-256 PoW solver
│   └── cpu_benchmark.py              # Direct single-core wall-clock / CPU utilization benchmark
├── scripts/
│   ├── run_all_experiments.py        # Master 30-trial Monte Carlo suite + statistical tests
│   ├── verify_proofs.py              # Numerical verification of Properties/Bounds 1–5 & pilot studies
│   ├── generate_figures.py           # Generates Figures 2–5 from JSON results (and copies Fig 1)
│   ├── summarize_for_paper.py        # Summarizes JSON metrics & statistics for manuscript tables
│   ├── audit_dois.py                 # Verifies CrossRef metadata for cited DOIs
│   └── audit_tex.py                  # Strict LaTeX structural, math & citation auditor
├── mininet_testbed/
│   ├── topology_wan.py               # Optional unexecuted Mininet/OVS WAN topology template (not cited as evidence)
│   ├── Dockerfile                    # Containerized reproducible execution environment
│   └── run_mininet.sh                # Optional Linux kernel runner script (not executed in manuscript evaluation)
├── tests/
│   └── test_aobctrm_pow.py           # Unit tests verifying mathematical invariants & edge cases
└── results/
    ├── all_experiment_results.json   # Full summary metrics, statistics & distributions
    ├── trial_telemetry.csv           # Per-trial telemetry across all 30 Monte Carlo runs
    ├── proof_verification.json       # Property, bound, probation & bandit pilot verification results
    └── figures/                      # Figure 1 (architecture diagram) & generated 300 DPI Figs 2–5
```

---

## Quick Start & Full Reproduction

### 1. Install Dependencies
Exact bit-for-bit reproduction of RNG-coupled `scikit-learn` Random Forest splits and SciPy/NumPy arrays requires **Python 3.14.3** and the pinned lockfile:
```bash
pip install -r requirements-lock.txt
```
Minimum unpinned package versions are also listed in `requirements.txt` (hardware timing results in CPU benchmarks vary by host processor and OS scheduling).

### 2. Run Analytical Property Checks & Automated Unit Tests
```bash
python -m unittest discover -s tests -v
python scripts/verify_proofs.py
```

### 3. Run All 30 Monte Carlo Simulation Trials & Statistical Tests
```bash
python scripts/run_all_experiments.py
```
Outputs raw CSV/JSON simulated telemetry into `results/trial_telemetry.csv` and `results/all_experiment_results.json`.

### 4. Regenerate Figures Directly from Simulation Results
```bash
python scripts/generate_figures.py
```

### 5. Optional Linux Topology Scripts (Not Cited as Manuscript Evidence)
The `mininet_testbed/` directory retains optional, unexecuted Linux Mininet/OVS topology templates. The 50-node WAN sensitivity results reported in the manuscript are generated solely from the Python discrete-time analytic model (`sim/consensus_sim.py`), and `mininet_testbed/` is not cited as validation evidence.

---

## Verified Key Results (30-Trial Monte Carlo Suite, Seeds `100..129`)

*Model scope:* Discrete-time (`120 s`) consensus/mempool model with closed-form analytic BIP-152 propagation (`B_prop`), decision latency (`B_cr`), fork rate, and bandwidth equations. CPU percentages are hardware-specific wall-clock measurements normalized to a `92.0 ms` single-core slot budget on **Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz** (with symmetric `1x` per-slot execution for all defenses). `Standard PoW`, `Token Bucket`, `EWMA`, and `Dynamic Fee` share essentially identical CPU means (`14.9–15.0%`) and identical `0.6%` SD (`14.95–14.97 ± 0.57%`) because `sim/cpu_benchmark.py` adds each method's tiny per-slot incremental overhead (`< 0.03 ms`) to a single shared 20-trial SHA-256 validation baseline array (`13.76 ± 0.53 ms`). Every admission filter's DDoS spam leakage is driven directly by its own emergent FNR (`(admitted & is_mal).sum() / n_mal`) and uniform honest-relay loss weight (`0.20`). Token Bucket uses `token_bucket_peers = 55` (`B=500`, `r=50 tx/s`), intentionally calibrated to a `2,970 tx/s` aggregate refill cap (bracketed between `2,850 tx/s` nominal honest demand and `3,400 tx/s` nominal service capacity) so that congestion throttling binds realistically under the `5,000 tx/s` flood. Bandwidth `± 0.0` denotes `SD < 0.05 MB/s` rounded to one decimal place.

| Model (`30%` Traitors, `5,000 tx/s` DDoS) | $B_{cr}$ (s) | $B_{\text{prop}}$ (s) | Single-Core CPU (%) | Fork Rate (%) | Bandwidth (MB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard PoW (Undefended) | $13.50 \pm 0.10$ | $5.42 \pm 0.06$ | $14.9 \pm 0.6$ | $10.93 \pm 0.09$ | $19.6 \pm 0.0$ |
| ML-IDS (100-tree RF) | $2.09 \pm 0.01$ | $0.44 \pm 0.00$ | $30.9 \pm 1.0$ | $1.06 \pm 0.01$ | $11.0 \pm 0.0$ |
| Static TOPSIS | $2.13 \pm 0.04$ | $0.47 \pm 0.02$ | $15.4 \pm 0.6$ | $1.17 \pm 0.05$ | $11.1 \pm 0.1$ |
| EWMA Reputation | $2.44 \pm 0.04$ | $0.60 \pm 0.02$ | $15.0 \pm 0.6$ | $1.39 \pm 0.03$ | $11.2 \pm 0.0$ |
| Token Bucket (`2,970 tx/s` cap) | $5.46 \pm 0.03$ | $2.14 \pm 0.02$ | $14.9 \pm 0.6$ | $6.01 \pm 0.03$ | $12.4 \pm 0.0$ |
| Dynamic Fee | $13.12 \pm 0.10$ | $5.28 \pm 0.06$ | $15.0 \pm 0.6$ | $10.72 \pm 0.09$ | $15.0 \pm 0.0$ |
| **A-OBCTrm-PoW (Proposed)** | **$2.07 \pm 0.01$** | **$0.44 \pm 0.00$** | **$15.9 \pm 0.6$** | **$1.06 \pm 0.01$** | **$10.9 \pm 0.0$** |

- **Modeled Breakdown Boundary (`f = 0.50`, not a formal BFT guarantee):** Under reliable peer feedback in `seeds 100..129`, A-OBCTrm-PoW filters `99.6%` (`FNR = 0.38 ± 0.28%`, `FPR = 4.38 ± 0.99%`, `f_eff = 0.0040 ± 0.0029`, `100.0%` pool uptime). Under compromised majority aged feedback, filtering collapses to `35.5%` (`FNR = 64.47 ± 4.59%`, `FPR = 9.20 ± 3.10%`, `f_eff = 0.4148 ± 0.0113 > 1/3`, `0.0%` uptime; unseen `seeds 200..229` yield `62.52%` FNR and `f_eff ≈ 0.41`), confirming that `f > 1/3` resilience is strictly contingent on measurement integrity ($\psi \le 1 - \rho$) and does not establish a formal BFT guarantee.
- **Automated Verification:** `python -m unittest discover -s tests -v` (`8` tests passing) and `python scripts/verify_proofs.py` verify Properties & Bounds 1–5 against exact numerical outputs in `results/proof_verification.json`.
