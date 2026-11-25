# Analytical Properties, Algebraic Bounds, and Model Checks for A-OBCTrm-PoW

This document provides self-contained analytical derivations of the algebraic bounds, trust dynamics, and scoped pool-weight properties of **A-OBCTrm-PoW** (`sim/trust_engine.py`), clearly separated from their discrete-time simulation checks (`scripts/verify_proofs.py` and `tests/test_aobctrm_pow.py`). These are mechanism properties and conditional algebraic identities—not formal distributed-systems protocol security proofs.

---

## Property 1: Smoothness, Monotonicity, and Bounded Simplex Sharing of Adaptive Weights

**Statement.**  
Let $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0, 1]^3$ be the normalized latency, transaction-rate pressure, and anomaly-intensity metrics. Define the pre-normalized driving functions with positive sensitivity coefficients $k_1 = 2.5$, $k_2 = 1.5$, $k_3 = 1.0$:
$$f_{\mathrm{THS}}(\hat{\delta}) = e^{k_1 \hat{\delta}}, \qquad f_{\mathrm{NSI}}(\hat{\lambda}) = 1 + k_2 \hat{\lambda}, \qquad f_{\mathrm{PRF}}(\hat{\tau}) = 1 + k_3 \hat{\tau},$$
with normalized weight vector $\tilde{w}_j = \frac{f_j}{S}$ where $S = f_{\mathrm{THS}}(\hat{\delta}) + f_{\mathrm{NSI}}(\hat{\lambda}) + f_{\mathrm{PRF}}(\hat{\tau})$, followed by upper simplex clamping at $w_{\max} = 0.80$ when $\tilde{w}_{\mathrm{THS}} > 0.80$ (redistributing $\tilde{w}_{\mathrm{THS}} - 0.80$ proportionally across NSI and PRF). Then:
1. *(Simplex Validity):* $w_j \in (0, 0.80]$ and $\sum_{j=1}^3 w_j = 1$ for every $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0, 1]^3$.
2. *(Strict Monotonicity):* In the unclamped operating region ($\tilde{w}_{\mathrm{THS}} < 0.80$), $\frac{\partial w_{\mathrm{THS}}}{\partial \hat{\delta}} > 0$ strictly everywhere. Under joint attack stress $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) = (1, 1, 1)$, $\tilde{w}_{\mathrm{THS}} = \frac{e^{2.5}}{e^{2.5} + 2.5 + 2.0} \approx 0.7303 < 0.80$, whereas isolated pure-anomaly stress $(0, 0, 1)$ attains $\tilde{w}_{\mathrm{THS}} = \frac{e^{2.5}}{e^{2.5} + 2} \approx 0.8590$, which is clamped to $0.80$.
3. *(Lipschitz Continuity in $\ell_2$ Norm):* The scalar THS component satisfies $\left|\frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}}\right| \le \frac{k_1}{4} = 0.625$ everywhere ($\left|\frac{\partial w_{\mathrm{THS}}}{\partial \hat{\delta}}\right| \le 0.625$ a.e. after clamping). Under Euclidean ($\ell_2$) norm over the full 3-component weight vector $\mathbf{w}(\hat{\lambda}, \hat{\tau}, \hat{\delta}) = [w_{\mathrm{THS}}, w_{\mathrm{NSI}}, w_{\mathrm{PRF}}]^\top \in \mathbb{R}^3$, $\mathbf{w}$ is globally Lipschitz-continuous on $[0,1]^3$, with directional $\ell_2$ constant $\sup \|\partial \mathbf{w} / \partial \hat{\delta}\|_2 \approx 0.788$ (`0.787591` at step $0.05$, `0.788326` at step $0.025$) and full 3D Euclidean $\ell_2$ Jacobian spectral-norm constant $\approx 0.813$--$0.814$ (`0.812994` at step $0.05$, `0.813944` at step $0.025$).

**Proof.**  
1. Since $f_j \ge 1$ for all $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0,1]^3$ (attaining equality at $(0,0,0)$), $S \ge 1 + 1 + 1 = 3$, giving $\tilde{w}_j > 0$ and $\sum_j \tilde{w}_j = 1$. Whenever $\tilde{w}_{\mathrm{THS}} > w_{\max} = 0.80$, we have $\tilde{w}_{\mathrm{NSI}} + \tilde{w}_{\mathrm{PRF}} = 1 - \tilde{w}_{\mathrm{THS}} < 0.20$; redistribution sets $w_{\mathrm{THS}} = 0.80$ and scales $\tilde{w}_{\mathrm{NSI}}, \tilde{w}_{\mathrm{PRF}}$ by $\frac{1 - w_{\max}}{1 - \tilde{w}_{\mathrm{THS}}} = \frac{0.20}{1 - \tilde{w}_{\mathrm{THS}}} > 0$, ensuring $w_{\mathrm{NSI}} + w_{\mathrm{PRF}} = 0.20$ with each component strictly $< 0.20 < 0.80$. Thus $w_j \in (0, 0.80]$ and $\sum_j w_j = 1$ always.
2. Differentiating $\tilde{w}_{\mathrm{THS}}$ with respect to $\hat{\delta}$ holding $\hat{\lambda}, \hat{\tau}$ fixed (let $B = f_{\mathrm{NSI}} + f_{\mathrm{PRF}} \ge 2$):
   $$\frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}} = \frac{k_1 e^{k_1\hat{\delta}} (e^{k_1\hat{\delta}} + B) - e^{k_1\hat{\delta}}(k_1 e^{k_1\hat{\delta}})}{(e^{k_1\hat{\delta}} + B)^2} = \frac{k_1 B e^{k_1\hat{\delta}}}{(e^{k_1\hat{\delta}} + B)^2} > 0.$$
   Directly substituting $(1, 1, 1)$ gives $e^{2.5} \approx 12.1825$, $B = 2.5 + 2.0 = 4.5$, and $12.1825 / 16.6825 \approx 0.73026 < 0.80$. At $(0, 0, 1)$, $B = 2.0$, giving $12.1825 / 14.1825 \approx 0.85898 > 0.80$, requiring explicit simplex clamping at $w_{\max} = 0.80$.
3. By AM--GM on $x = e^{k_1\hat{\delta}} > 0$ and $B > 0$, $(x + B)^2 \ge 4 x B$, hence for the scalar THS component:
   $$0 < \frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}} = k_1 \frac{x B}{(x+B)^2} \le \frac{k_1}{4} = \frac{2.5}{4} = 0.625.$$
   Clamping $w_{\mathrm{THS}} = \min(\tilde{w}_{\mathrm{THS}}, w_{\max})$ preserves $|\partial w_{\mathrm{THS}} / \partial \hat{\delta}| \le 0.625$ almost everywhere. Furthermore, since $f_{\mathrm{NSI}}, f_{\mathrm{PRF}} \ge 1$ and $f_{\mathrm{THS}} \le e^{k_1}$, we always have $1 - \tilde{w}_{\mathrm{THS}} \ge \frac{2}{e^{2.5}+2} \approx 0.141 > 0$; thus proportional redistribution for $\tilde{w}_{\mathrm{THS}} > w_{\max}$ is continuous across the clamping boundary $\tilde{w}_{\mathrm{THS}} = w_{\max}$ and smooth on both sides with bounded Jacobian entries everywhere. Evaluating the full 3-component vector $\mathbf{w}$ under Euclidean ($\ell_2$) norm over $[0,1]^3$ confirms $\|\Delta \mathbf{w}\|_2 / \Delta \hat{\delta} \le 0.788$ (and spectral norm $\|\nabla \mathbf{w}\|_2 \le 0.814$), preventing discontinuous weight jumps under telemetry noise. $\blacksquare$

---

## Property 2: Asymmetric Trust Penalty and Closed-Form Bootstrapping Recovery Horizon

**Statement.**  
Under the trust update rule
$$\mathrm{THS}_i(t+1) = \begin{cases} \min\bigl(1.0,\; \mathrm{THS}_i(t) + \eta [1 - \mathrm{THS}_i(t)]\bigr), & \phi_i(t) = 1, \\ \max\bigl(0.0,\; \kappa \mathrm{THS}_i(t) - \zeta\bigr), & \phi_i(t) = 0, \end{cases}$$
with $\eta \in (0, 1)$, $\kappa \in (0, 1)$, and $\zeta > 0$ ($\eta = 0.05, \kappa = 0.40, \zeta = 0.05$):
1. *(Immediate Collapse):* A single detected defection reduces any non-zero trust score $T_0 > 0$ by strictly more than $(1 - \kappa) = 60\%$ of $T_0$. Specifically, fractional loss is $1 - \max(0, \kappa - \zeta / T_0) > 60\%$.
2. *(Exact Honest Recovery Horizon):* From any post-penalty score $T_{\mathrm{post}} \in [0, T_{\mathrm{target}})$, recovering to any admission threshold $T_{\mathrm{target}} \in (T_{\mathrm{post}}, 1)$ requires exactly
   $$k_{\mathrm{rec}}(T_{\mathrm{post}} \to T_{\mathrm{target}}) = \left\lceil \frac{\ln\!\left(\frac{1 - T_{\mathrm{target}}}{1 - T_{\mathrm{post}}}\right)}{\ln(1 - \eta)} \right\rceil$$
   consecutive valid rounds ($\phi_i = 1$). In particular, full recovery from $T_{\mathrm{post}} = 0$ to $0.50$ requires $14$ rounds, and recovery to $0.75$ requires $28$ rounds (and $20$--$24$ rounds following single-step defection from $T_0 \in [0.50, 0.95]$).

**Proof.**  
1. If $\kappa T_0 - \zeta \le 0$, trust drops $100\%$ to $0$. Otherwise $\mathrm{THS}_{\mathrm{post}} = \kappa T_0 - \zeta < \kappa T_0 = 0.40\,T_0$, giving fractional loss $\frac{T_0 - \mathrm{THS}_{\mathrm{post}}}{T_0} = (1 - \kappa) + \frac{\zeta}{T_0} > 0.60$.
2. Define error gap $g(t) = 1 - \mathrm{THS}_i(t)$. Along consecutive valid steps ($\phi_i = 1$), since $\mathrm{THS}_i(t) < 1$ and $\eta < 1$, we have $\mathrm{THS}_i(t) + \eta(1 - \mathrm{THS}_i(t)) < 1$, so the update is exact linear contraction:
   $$g(t+1) = 1 - \mathrm{THS}_i(t) - \eta g(t) = (1 - \eta) g(t).$$
   By induction over $k \ge 0$ valid rounds, $1 - \mathrm{THS}_i(t+k) = (1 - \eta)^k (1 - T_{\mathrm{post}})$. Requiring $\mathrm{THS}_i(t+k) \ge T_{\mathrm{target}}$ is equivalent to $(1 - \eta)^k \le \frac{1 - T_{\mathrm{target}}}{1 - T_{\mathrm{post}}}$. Taking logarithms (noting $\ln(1 - \eta) < 0$ reverses the inequality) gives $k \ge \frac{\ln((1 - T_{\mathrm{target}})/(1 - T_{\mathrm{post}}))}{\ln(1 - \eta)}$. Taking the ceiling yields the exact minimum integer $k_{\mathrm{rec}}$. Setting $T_{\mathrm{post}} = 0, T_{\mathrm{target}} = 0.50, \eta = 0.05$ yields $\lceil \ln(0.5)/\ln(0.95)\rceil = \lceil 13.513\rceil = 14$. $\blacksquare$

---

## Bound 3: Algebraic Credibility-Weighted PRF Perturbation Bound

**Statement.**  
Let honest peers $h \in \mathcal{H}$ possess credibility sum $C_{\mathcal{H}} = \sum_{h \in \mathcal{H}} c_h > 0$, where $c_k = \min(1, \mathrm{THS}_k / \theta_c)$. Suppose an adversary introduces $s$ Sybil identities $\mathcal{S}$ with mean credibility $\bar{c}_{\mathcal{S}} = \frac{1}{s}\sum_{k \in \mathcal{S}} c_k$, all submitting arbitrary feedback $r_{k \to i} \in [0, 1]$. For any target node $i$ whose honest-only credibility-weighted PRF is $\mathrm{PRF}_i^{\mathcal{H}} = \frac{\sum_{h \in \mathcal{H}} c_h r_{h \to i}}{C_{\mathcal{H}} + \varepsilon}$:
1. *(Zero-History Sybil Weight Exclusion):* Newly created Sybils with zero validation history ($\mathrm{THS}_k = 0 \implies c_k = 0$) induce zero perturbation ($\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} = \mathrm{PRF}_i^{\mathcal{H}}$).
2. *(Worst-Case Perturbation Upper Bound):* For arbitrary Sybil credibility $\bar{c}_{\mathcal{S}} \ge 0$, the absolute PRF shift satisfies the algebraic upper bound:
   $$\left| \mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}} \right| \le \frac{s\,\bar{c}_{\mathcal{S}}}{s\,\bar{c}_{\mathcal{S}} + C_{\mathcal{H}} + \varepsilon}.$$
3. *(Weight-Share Breakdown Condition):* The perturbation upper bound remains small only while $s\,\bar{c}_{\mathcal{S}} \ll C_{\mathcal{H}}$. Once a coalition accumulates enough aged identities or compromises enough high-history voters that $s\,\bar{c}_{\mathcal{S}} \ge C_{\mathcal{H}}$, the adversarial credibility-weight share exceeds $1/2$ (up to regularization $\varepsilon$) and the worst-case PRF perturbation upper bound reaches $1/2$ or higher (with actual score shift scaling proportionally to $|r_{\mathcal{S}} - r_{\mathcal{H}}|$).

**Proof.**  
Let $N_{\mathcal{H}} = \sum_{h \in \mathcal{H}} c_h r_{h \to i}$ and $N_{\mathcal{S}} = \sum_{k \in \mathcal{S}} c_k r_{k \to i} \in [0, s\bar{c}_{\mathcal{S}}]$. By definition of $c_k$, zero-history Sybils contribute $0$ to both numerator and denominator, proving (1). In general:
$$\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}} = \frac{N_{\mathcal{H}} + N_{\mathcal{S}}}{C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon} - \frac{N_{\mathcal{H}}}{C_{\mathcal{H}} + \varepsilon} = \frac{N_{\mathcal{S}}(C_{\mathcal{H}} + \varepsilon) - s\bar{c}_{\mathcal{S}} N_{\mathcal{H}}}{(C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon)(C_{\mathcal{H}} + \varepsilon)}.$$
Since $0 \le \frac{N_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}}} \le 1$ and $0 \le \frac{N_{\mathcal{H}}}{C_{\mathcal{H}}+\varepsilon} \le 1$ (assuming $s\bar{c}_{\mathcal{S}} > 0$), factoring out the weight share and taking absolute values gives:
$$\left|\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}}\right| = \frac{s\bar{c}_{\mathcal{S}}}{C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon} \left|\frac{N_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}}} - \frac{N_{\mathcal{H}}}{C_{\mathcal{H}} + \varepsilon}\right| \le \frac{s\bar{c}_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}} + C_{\mathcal{H}} + \varepsilon}.$$
When $s\bar{c}_{\mathcal{S}} \ge C_{\mathcal{H}} + \varepsilon$, the weight-share upper bound reaches $\ge 1/2$, confirming loss of honest credibility majority (note this is an algebraic perturbation bound, not a cryptographic proof against collusion). $\blacksquare$

---

## Property 4: Algebraic Pool-Weight Ratio Under Assumed Classifier Performance ($f < 1/3$)

**Statement (Algebraic Identity & Operational Assumption—Not a Formal Byzantine Safety Proof).**  
Let total network mining weight be normalized to $1$, with adversarial weight fraction $f \in [0, 1)$ and honest weight $1 - f$. Denote the realized honest false-exclusion rate (FPR) by $\rho \in [0, 1)$ and malicious admission rate (FNR) by $\psi \in [0, 1]$. Then:
1. *(Retained Pool Weights):* By set partitioning, the active High-Reliability Miner (HM) pool holds honest weight $H_{\mathrm{HM}} = (1 - f)(1 - \rho)$ and malicious weight $A_{\mathrm{HM}} = f\psi$.
2. *(Effective Post-Filter Share):* Conditional on $(\rho, \psi)$, the adversarial share inside $\mathcal{M}_{\mathrm{HM}}$ is the algebraic ratio:
   $$f_{\mathrm{eff}} = \frac{f\psi}{(1 - f)(1 - \rho) + f\psi}.$$
3. *(Conditional $f < 1/3$ Scope Preservation):* Under the explicit operational assumption that honest recall weakly dominates malicious bypass ($1 - \rho \ge \psi$), $f < 1/3$ implies $f_{\mathrm{eff}} < 1/3$ strictly. Conversely, when $f \ge 1/3$ or when $\psi > 1 - \rho$, no Byzantine fault tolerance or liveness bound holds; $(\rho, \psi)$ are outcomes of trust measurements whose integrity degrades when feedback majorities are compromised.

**Proof.**  
By disjoint set partitioning over $\mathcal{H} \sqcup \mathcal{A}$: $|\mathcal{H} \cap \mathcal{M}_{\mathrm{HM}}|/m = (1-f)(1-\rho)$ and $|\mathcal{A} \cap \mathcal{M}_{\mathrm{HM}}|/m = f\psi$, whose ratio defines $f_{\mathrm{eff}}$. Checking $f_{\mathrm{eff}} < 1/3$ yields $3f\psi < (1-f)(1-\rho) + f\psi \iff 2f\psi < (1-f)(1-\rho)$. Assuming $f < 1/3$ (so $1 - f > 2/3$) and $\psi \le 1 - \rho$:
$$2 f\psi < \frac{2}{3}(1 - \rho) < (1 - f)(1 - \rho).$$
Because $\psi \le 1 - \rho$ is an assumed property of the underlying classifier metrics rather than a guaranteed invariant under adversarial feedback poisoning, this property is a conditional algebraic bound—not a formal Byzantine security guarantee. $\blacksquare$

**Modeled Numerical Illustration (Simulation Outputs, Seeds `100..129`, $n = 30$, Mean $\pm$ SD):**
- *Reliable feedback ($f = 0.50$):* $\rho = 0.0438 \pm 0.0099$, $\psi = 0.0038 \pm 0.0028$, yielding $f_{\mathrm{eff}} = 0.0040 \pm 0.0029 < 1/3$.
- *Corrupted aged majority feedback ($f = 0.50$):* $\rho = 0.0920 \pm 0.0310$, $\psi = 0.6447 \pm 0.0459$, yielding $f_{\mathrm{eff}} = 0.4148 \pm 0.0113 > 1/3$ (and across 30 unseen seeds `200..229`, mean $\psi = 0.6252 \implies f_{\mathrm{eff}} \approx 0.41 > 1/3$), consistently violating $f_{\mathrm{eff}} < 1/3$.

---

## Property 5: Finite-Sample Breakdown Boundedness of Trimmed Thresholding & Complexity

**Statement.**  
For $m$ miners and $n = 3$ criteria:
1. *(Finite-Sample Breakdown Boundedness vs. Tail-Rank Invariance):* Let $\alpha_{\mathrm{trim}} \in (0, 0.5)$ denote the symmetric per-tail trimming fraction in Eq. (12) ($k = \lfloor \alpha_{\mathrm{trim}} m \rfloor$ discarded from each tail). In the standard robust-statistics finite-sample breakdown sense, $\mu_{\mathrm{trim}}$ and $\sigma_{\mathrm{trim}}$ have breakdown point $\epsilon^* = (k + 1)/m \approx \alpha_{\mathrm{trim}}$ per tail: arbitrary corruption or replacement of up to $k$ entries leaves $\mu_{\mathrm{trim}}$ and $\sigma_{\mathrm{trim}}$ strictly **bounded** within the uncorrupted central order-statistic interval rather than diverging under arbitrary $\pm\infty$ outliers (note that exact `0.00%` invariance requires the stronger condition that all replaced entries occupy the outermost $\le k$ tail ranks without displacing middle order statistics).
2. *(Algorithmic Complexity):* Algorithm 1 requires exact worst-case $O(m \cdot n)$ arithmetic time for TOPSIS scoring and $O(m)$ expected time via Quickselect (`np.partition`) for threshold computation (or $O(m \log m)$ via full sort), i.e., $O(1)$ work per miner for fixed $n = 3$ and $O(m)$ overall time.

**Proof.**  
Order closeness coefficients $C_{(1)} \le \dots \le C_{(m)}$ and discard $k = \lfloor \alpha_{\mathrm{trim}} m \rfloor$ elements from each end. If at most $k$ entries anywhere in the sample are replaced by arbitrary extreme outliers, at least one extreme tail discards those corrupted values when placed at the tails, and if placed inside, at most $k$ original interior scores shift across the cut boundaries—leaving $\mu_{\mathrm{trim}}$ and $\sigma_{\mathrm{trim}}$ strictly **bounded** by the extreme non-corrupted interior ranks rather than exploding. Exact invariance (`0.00%` shift) holds strictly in the special case where the $\le k$ modified values lie exclusively among the lowest $k$ (or highest $k$) sorted positions. For complexity, column norming, scaling, $A^+/A^-$ scans, and Euclidean distances traverse $m \times n$ arrays in $O(m \cdot n)$ time, while Quickselect finds order statistics $k$ and $m - k$ in $O(m)$ expected time. $\blacksquare$

**Simulation Tail-Replacement Check (Fixed Cohort $m = 1{,}000$, `scripts/verify_proofs.py` — Numerical Check, Not Part of Proof):**  
Replacing the lowest $5\%$ (`50`) tail ranks on a synthetic $1{,}000$-miner sample (`baseline T = 0.4979`) yields `0.00%` shift in $\mu_{\mathrm{trim}}$ and $T(t)$, whereas replacing $10\%$ and $20\%$ lowest-tail entries shifts $\mu_{\mathrm{trim}}$ by `1.22%` and `4.61%` (with $T(t)$ clamped at $T_{\min} = 0.47$, shifting `4.31%` and `5.61%`).
