# A-OBCTrm-PoW: Replication Package, Simulation Suite & Formal Proofs

This repository provides the full reproducible codebase, formal mathematical proofs, automated verification tests, Docker/Mininet testbed configurations, and raw experimental telemetry for the manuscript:

> **Lightweight Trust-Aware Consensus Optimization for Efficient Blockchain Performance Under Adversarial Conditions**  
> Puppala Naga Sravanthi and Pasam Naveen  
> Department of Computer Science and Engineering, School of Engineering and Applied Sciences, SRM University-AP, Amaravati, 522240, India

---

## Repository Structure

```text
A-OBCTrm-PoW-Replication/
├── PROOFS.md                         # Formal mathematical derivations (Theorems 1–5)
├── CITATION.cff                      # Software citation metadata
├── LICENSE                           # MIT License
├── requirements.txt                  # Python package dependencies
├── sim/
│   ├── config.py                     # All consensus, TOPSIS, threshold & topology hyperparameters
│   ├── topology.py                   # 1,000-node SimBlock-calibrated overlay + 50-node WAN graph
│   ├── trust_engine.py               # Dynamic TOPSIS, Credibility PRF, asymmetric THS & probation
│   ├── baselines.py                  # Standard PoW, 100-tree RF ML-IDS, Static TOPSIS, EWMA, Token Bucket, Dynamic Fee
│   ├── consensus_sim.py              # Discrete-event PoW consensus & adversarial simulator + SHA-256 PoW solver
│   └── cpu_benchmark.py              # Direct single-core wall-clock / CPU utilization benchmark
├── scripts/
│   ├── run_all_experiments.py        # Master 30-trial Monte Carlo suite + statistical tests
│   ├── verify_proofs.py              # Numerical verification of Theorems 1–5 & pilot studies
│   ├── generate_figures.py           # Generates Figures 2, 3, 4, and 5 from JSON results
│   ├── summarize_for_paper.py        # Summarizes JSON metrics & statistics for manuscript tables
│   ├── audit_dois.py                 # Verifies CrossRef metadata for cited DOIs
│   └── audit_tex.py                  # Strict LaTeX structural, math & citation auditor
├── mininet_testbed/
│   ├── topology_wan.py               # Linux Mininet + OVS + tc netem 50-node WAN topology
│   ├── Dockerfile                    # Containerized reproducible execution environment
│   └── run_mininet.sh                # Runner script for Linux kernel hosts
├── tests/
│   └── test_aobctrm_pow.py           # Unit tests verifying mathematical invariants & edge cases
└── results/
    ├── all_experiment_results.json   # Full summary metrics, statistics & distributions
    ├── trial_telemetry.csv           # Per-trial telemetry across all 30 Monte Carlo runs
    ├── proof_verification.json       # Theorem, probation & bandit pilot verification results
    └── figures/                      # Publication-ready 300 DPI PNG figures (Figures 1–5)
```

---

## Quick Start & Full Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Formal Mathematical Proofs & Automated Unit Tests
```bash
python -m unittest discover -s tests -v
python scripts/verify_proofs.py
```

### 3. Run All 30 Monte Carlo Simulation Trials & Statistical Tests
```bash
python scripts/run_all_experiments.py
```
Outputs raw CSV/JSON telemetry into `results/trial_telemetry.csv` and `results/all_experiment_results.json`.

### 4. Regenerate Figures Directly from Experimental Results
```bash
python scripts/generate_figures.py
```

### 5. Linux Mininet + OVS Kernel Testbed (Optional, Requires Privileged Linux Kernel)
```bash
bash mininet_testbed/run_mininet.sh
```

---

## Verified Key Results (30-Trial Monte Carlo Suite, Seeds `100..129`)

| Model (`30%` Traitors, `5,000 tx/s` DDoS) | $B_{cr}$ (s) | $B_{\text{prop}}$ (s) | Single-Core CPU (%) | Fork Rate (%) | Bandwidth (MB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard PoW (Undefended) | $13.50 \pm 0.10$ | $5.42 \pm 0.06$ | $12.9 \pm 1.9$ | $10.93 \pm 0.09$ | $19.6 \pm 0.0$ |
| ML-IDS (100-tree RF) | $10.45 \pm 0.03$ | $2.83 \pm 0.03$ | $70.6 \pm 4.5$ | $4.65 \pm 0.04$ | $12.6 \pm 0.0$ |
| Static TOPSIS | $7.77 \pm 0.05$ | $2.38 \pm 0.04$ | $13.4 \pm 1.9$ | $4.03 \pm 0.08$ | $13.4 \pm 0.0$ |
| EWMA Reputation | $9.06 \pm 0.05$ | $2.82 \pm 0.04$ | $12.9 \pm 1.9$ | $4.71 \pm 0.05$ | $14.7 \pm 0.0$ |
| Token Bucket | $13.49 \pm 0.10$ | $5.41 \pm 0.06$ | $12.9 \pm 1.9$ | $10.92 \pm 0.09$ | $18.5 \pm 0.0$ |
| Dynamic Fee | $13.12 \pm 0.10$ | $5.28 \pm 0.06$ | $12.9 \pm 1.9$ | $10.72 \pm 0.09$ | $15.0 \pm 0.0$ |
| **A-OBCTrm-PoW (Proposed)** | **$2.47 \pm 0.01$** | **$0.44 \pm 0.00$** | **$13.8 \pm 1.9$** | **$1.06 \pm 0.01$** | **$10.9 \pm 0.0$** |

- **Empirical Breakdown Boundary (`f = 0.50`):** Under reliable peer feedback, A-OBCTrm-PoW filters `99.6%` (`FNR = 0.38%`, `100.0%` pool uptime). Under compromised majority aged feedback, filtering collapses to `35.5%` (`FNR = 64.47%`, `0.0%` uptime), confirming that `f > 1/3` resilience is strictly contingent on measurement integrity and does not establish a formal BFT guarantee.
- **Automated Verification:** `python -m unittest discover -s tests -v` (`6` tests passing in `<1s`) and `python scripts/verify_proofs.py` verify Theorems 1–5 against exact numerical outputs in `results/proof_verification.json`.
