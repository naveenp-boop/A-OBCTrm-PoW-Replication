# Formal Mathematical Proofs and Mechanistic Guarantees for A-OBCTrm-PoW

This document provides complete, self-contained formal proofs supporting the core algorithmic properties, stability bounds, and scoped liveness guarantees of **A-OBCTrm-PoW** (`sim/trust_engine.py`), along with automated numerical verification (`scripts/verify_proofs.py` and `tests/test_aobctrm_pow.py`).

---

## Theorem 1: Smoothness, Monotonicity, and Bounded Simplex Sharing of Adaptive Weights

**Statement.**  
Let $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0, 1]^3$ be the normalized latency, transaction-rate pressure, and anomaly-intensity metrics. Define the pre-normalized driving functions with positive sensitivity coefficients $k_1 = 2.5$, $k_2 = 1.5$, $k_3 = 1.0$:
$$f_{\mathrm{THS}}(\hat{\delta}) = e^{k_1 \hat{\delta}}, \qquad f_{\mathrm{NSI}}(\hat{\lambda}) = 1 + k_2 \hat{\lambda}, \qquad f_{\mathrm{PRF}}(\hat{\tau}) = 1 + k_3 \hat{\tau},$$
with normalized weight vector $\tilde{w}_j = \frac{f_j}{S}$ where $S = f_{\mathrm{THS}}(\hat{\delta}) + f_{\mathrm{NSI}}(\hat{\lambda}) + f_{\mathrm{PRF}}(\hat{\tau})$, followed by upper simplex clamping at $w_{\max} = 0.80$ when $\tilde{w}_{\mathrm{THS}} > 0.80$ (redistributing $\tilde{w}_{\mathrm{THS}} - 0.80$ proportionally across NSI and PRF). Then:
1. *(Simplex Validity):* $w_j \in (0, 0.80]$ and $\sum_{j=1}^3 w_j = 1$ for every $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0, 1]^3$.
2. *(Strict Monotonicity):* In the unclamped operating region ($\tilde{w}_{\mathrm{THS}} < 0.80$), $\frac{\partial w_{\mathrm{THS}}}{\partial \hat{\delta}} > 0$ strictly everywhere. Under joint attack stress $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) = (1, 1, 1)$, $\tilde{w}_{\mathrm{THS}} = \frac{e^{2.5}}{e^{2.5} + 2.5 + 2.0} \approx 0.7303 < 0.80$, whereas isolated pure-anomaly stress $(0, 0, 1)$ attains $\tilde{w}_{\mathrm{THS}} = \frac{e^{2.5}}{e^{2.5} + 2} \approx 0.8590$, which is cleanly clamped to $0.80$.
3. *(Lipschitz Stability Against Telemetry Jitter):* $w_j(\hat{\lambda}, \hat{\tau}, \hat{\delta})$ is globally Lipschitz-continuous on $[0,1]^3$, with $\left|\frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}}\right| \le \frac{k_1}{4} = 0.625$.

**Proof.**  
1. Since $f_j \ge 1 > 0$ for all $(\hat{\lambda}, \hat{\tau}, \hat{\delta}) \in [0,1]^3$, $S > 2 + 1 = 3$, giving $\tilde{w}_j > 0$ and $\sum_j \tilde{w}_j = 1$. If $\tilde{w}_{\mathrm{THS}} > w_{\max} = 0.80$, redistribution assigns $w_{\mathrm{THS}} = 0.80$ and multiplies $\tilde{w}_{\mathrm{NSI}}, \tilde{w}_{\mathrm{PRF}}$ by $\frac{1 - w_{\max}}{\tilde{w}_{\mathrm{NSI}} + \tilde{w}_{\mathrm{PRF}}} = \frac{0.20}{1 - \tilde{w}_{\mathrm{THS}}} > 0$. Because $f_{\mathrm{NSI}}/S < \frac{3.5}{e^{2.5} + 3.5} < 0.23$ when $\tilde{w}_{\mathrm{THS}} > 0.80$, wait: specifically when $\tilde{w}_{\mathrm{THS}} > 0.80$, $\tilde{w}_{\mathrm{NSI}} + \tilde{w}_{\mathrm{PRF}} < 0.20$, and scaling each by $0.20 / (1 - \tilde{w}_{\mathrm{THS}})$ ensures $w_{\mathrm{NSI}} + w_{\mathrm{PRF}} = 0.20$, with each component strictly $< 0.20 < 0.80$. Thus $w_j \in (0, 0.80]$ and $\sum_j w_j = 1$ always.
2. Differentiating $\tilde{w}_{\mathrm{THS}}$ with respect to $\hat{\delta}$ holding $\hat{\lambda}, \hat{\tau}$ fixed (let $B = f_{\mathrm{NSI}} + f_{\mathrm{PRF}} \ge 2$):
   $$\frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}} = \frac{k_1 e^{k_1\hat{\delta}} (e^{k_1\hat{\delta}} + B) - e^{k_1\hat{\delta}}(k_1 e^{k_1\hat{\delta}})}{(e^{k_1\hat{\delta}} + B)^2} = \frac{k_1 B e^{k_1\hat{\delta}}}{(e^{k_1\hat{\delta}} + B)^2} > 0.$$
   Directly substituting $(1, 1, 1)$ gives $e^{2.5} \approx 12.1825$, $B = 2.5 + 2.0 = 4.5$, and $12.1825 / 16.6825 \approx 0.73026 < 0.80$. At $(0, 0, 1)$, $B = 2.0$, giving $12.1825 / 14.1825 \approx 0.85898 > 0.80$, proving why explicit simplex clamping enforces $w_j \le 0.80$ unconditionally across $[0,1]^3$.
3. By AM--GM on $x = e^{k_1\hat{\delta}} > 0$ and $B > 0$, $(x + B)^2 \ge 4 x B$, hence:
   $$0 < \frac{\partial \tilde{w}_{\mathrm{THS}}}{\partial \hat{\delta}} = k_1 \frac{x B}{(x+B)^2} \le \frac{k_1}{4} = \frac{2.5}{4} = 0.625.$$
   Projecting onto the simplex preserves 1-Lipschitzness, so small estimation noise $|\Delta \hat{\delta}|$ cannot produce discontinuous weight jumps. $\blacksquare$

---

## Theorem 2: Asymmetric Trust Penalty and Closed-Form Bootstrapping Recovery Cost

**Statement.**  
Under the trust update rule
$$\mathrm{THS}_i(t+1) = \begin{cases} \min\bigl(1.0,\; \mathrm{THS}_i(t) + \eta [1 - \mathrm{THS}_i(t)]\bigr), & \phi_i(t) = 1, \\ \max\bigl(0.0,\; \kappa \mathrm{THS}_i(t) - \zeta\bigr), & \phi_i(t) = 0, \end{cases}$$
with $\eta \in (0, 1)$, $\kappa \in (0, 1)$, and $\zeta > 0$ ($\eta = 0.05, \kappa = 0.40, \zeta = 0.05$):
1. *(Immediate Collapse):* A single detected defection reduces any non-zero trust score $T_0 > 0$ by strictly more than $(1 - \kappa) = 60\%$ of $T_0$. Specifically, fractional loss is $1 - \max(0, \kappa - \zeta / T_0) > 60\%$.
2. *(Exact Honest Recovery Horizon):* From any post-penalty score $T_{\mathrm{post}} \in [0, T_{\mathrm{target}})$, recovering to any admission threshold $T_{\mathrm{target}} \in (T_{\mathrm{post}}, 1)$ requires exactly
   $$k_{\mathrm{rec}}(T_{\mathrm{post}} \to T_{\mathrm{target}}) = \left\lceil \frac{\ln\!\left(\frac{1 - T_{\mathrm{target}}}{1 - T_{\mathrm{post}}}\right)}{\ln(1 - \eta)} \right\rceil$$
   consecutive valid rounds ($\phi_i = 1$). In particular, full recovery from $T_{\mathrm{post}} = 0$ to $0.50$ requires $14$ rounds, and recovery to $0.75$ requires $28$ rounds.

**Proof.**  
1. If $\kappa T_0 - \zeta \le 0$, trust collapses $100\%$ to $0$. Otherwise $\mathrm{THS}_{\mathrm{post}} = \kappa T_0 - \zeta < \kappa T_0 = 0.40\,T_0$, giving fractional loss $\frac{T_0 - \mathrm{THS}_{\mathrm{post}}}{T_0} = (1 - \kappa) + \frac{\zeta}{T_0} > 0.60$.
2. Define error gap $g(t) = 1 - \mathrm{THS}_i(t)$. Along consecutive valid steps ($\phi_i = 1$), since $\mathrm{THS}_i(t) < 1$ and $\eta < 1$, we have $\mathrm{THS}_i(t) + \eta(1 - \mathrm{THS}_i(t)) < 1$, so the update is exact linear contraction:
   $$g(t+1) = 1 - \mathrm{THS}_i(t) - \eta g(t) = (1 - \eta) g(t).$$
   By induction over $k \ge 0$ valid rounds, $1 - \mathrm{THS}_i(t+k) = (1 - \eta)^k (1 - T_{\mathrm{post}})$. Requiring $\mathrm{THS}_i(t+k) \ge T_{\mathrm{target}}$ is equivalent to $(1 - \eta)^k \le \frac{1 - T_{\mathrm{target}}}{1 - T_{\mathrm{post}}}$. Taking logarithms (noting $\ln(1 - \eta) < 0$ reverses the inequality) gives $k \ge \frac{\ln((1 - T_{\mathrm{target}})/(1 - T_{\mathrm{post}}))}{\ln(1 - \eta)}$. Taking the ceiling yields the exact minimum integer $k_{\mathrm{rec}}$. Setting $T_{\mathrm{post}} = 0, T_{\mathrm{target}} = 0.50, \eta = 0.05$ yields $\lceil \ln(0.5)/\ln(0.95)\rceil = \lceil 13.513\rceil = 14$. $\blacksquare$

---

## Theorem 3: Bounded Influence of Zero-History Sybil Cliques and Explicit Breakdown Condition

**Statement.**  
Let honest peers $h \in \mathcal{H}$ possess credibility sum $C_{\mathcal{H}} = \sum_{h \in \mathcal{H}} c_h > 0$, where $c_k = \min(1, \mathrm{THS}_k / \theta_c)$. Suppose an adversary introduces $s$ Sybil identities $\mathcal{S}$ with mean credibility $\bar{c}_{\mathcal{S}} = \frac{1}{s}\sum_{k \in \mathcal{S}} c_k$, all submitting arbitrary biased feedback $r_{k \to i} \in [0, 1]$. For any target node $i$ whose honest-only credibility-weighted PRF is $\mathrm{PRF}_i^{\mathcal{H}} = \frac{\sum_{h \in \mathcal{H}} c_h r_{h \to i}}{C_{\mathcal{H}} + \varepsilon}$:
1. *(Zero-History Sybil Immunity):* Newly created Sybils with zero validation history ($\mathrm{THS}_k = 0 \implies c_k = 0$) induce zero perturbation ($\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} = \mathrm{PRF}_i^{\mathcal{H}}$).
2. *(Worst-Case Perturbation Bound):* For arbitrary Sybil credibility $\bar{c}_{\mathcal{S}} \ge 0$, the absolute perturbation is strictly bounded by:
   $$\left| \mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}} \right| \le \frac{s\,\bar{c}_{\mathcal{S}}}{s\,\bar{c}_{\mathcal{S}} + C_{\mathcal{H}} + \varepsilon}.$$
3. *(Breakdown Condition Under Majority Corrupted Feedback):* The bound only remains small when $s\,\bar{c}_{\mathcal{S}} \ll C_{\mathcal{H}}$. Once a coalition accumulates enough aged identities or compromises enough high-history voters that $s\,\bar{c}_{\mathcal{S}} \ge C_{\mathcal{H}}$ (a condition feasible when adversarial participation $f > 1/3$ compromises feedback channels), adversarial influence exceeds $50\%$ of PRF weight and credibility filtering degrades.

**Proof.**  
Let $N_{\mathcal{H}} = \sum_{h \in \mathcal{H}} c_h r_{h \to i}$ and $N_{\mathcal{S}} = \sum_{k \in \mathcal{S}} c_k r_{k \to i} \in [0, s\bar{c}_{\mathcal{S}}]$. Since $r_{k \to i} = 0$ if $c_k = 0$, zero-history Sybils contribute $0$ to both numerator and denominator, proving (1). In general:
$$\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}} = \frac{N_{\mathcal{H}} + N_{\mathcal{S}}}{C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon} - \frac{N_{\mathcal{H}}}{C_{\mathcal{H}} + \varepsilon} = \frac{N_{\mathcal{S}}(C_{\mathcal{H}} + \varepsilon) - s\bar{c}_{\mathcal{S}} N_{\mathcal{H}}}{(C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon)(C_{\mathcal{H}} + \varepsilon)}.$$
Since $0 \le \frac{N_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}}} \le 1$ and $0 \le \frac{N_{\mathcal{H}}}{C_{\mathcal{H}}+\varepsilon} \le 1$, dividing and taking absolute value yields:
$$\left|\mathrm{PRF}_i^{\mathcal{H} \cup \mathcal{S}} - \mathrm{PRF}_i^{\mathcal{H}}\right| = \frac{s\bar{c}_{\mathcal{S}}}{C_{\mathcal{H}} + s\bar{c}_{\mathcal{S}} + \varepsilon} \left|\frac{N_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}}} - \frac{N_{\mathcal{H}}}{C_{\mathcal{H}} + \varepsilon}\right| \le \frac{s\bar{c}_{\mathcal{S}}}{s\bar{c}_{\mathcal{S}} + C_{\mathcal{H}} + \varepsilon}.$$
Setting $s\bar{c}_{\mathcal{S}} \ge C_{\mathcal{H}}$ yields an upper bound $\ge 1/2$, confirming the mechanistic breakdown condition when feedback credibility majority is lost. $\blacksquare$

---

## Theorem 4: Scoped Honest Pool Capacity and Byzantine Liveness Condition ($f < 1/3$)

**Statement.**  
Let total network mining weight be normalized to $1$, with adversarial weight fraction $f \in [0, 1)$ and honest weight $1 - f$. Suppose the admission threshold $T(t)$ rejects an honest fraction $\rho \in [0, 1)$ (FPR) and admits a malicious fraction $\psi \in [0, 1]$ (FNR). Then:
1. *(Retained Honest Capacity):* The active High-Reliability Miner (HM) pool retains exact honest weight $H_{\mathrm{HM}} = (1 - f)(1 - \rho)$ and malicious weight $A_{\mathrm{HM}} = f\psi$.
2. *(Effective Post-Filter Fraction):* Conditional on the measurement error rates $(\rho, \psi)$, the adversarial share inside $\mathcal{M}_{\mathrm{HM}}$ is:
   $$f_{\mathrm{eff}} = \frac{f\psi}{(1 - f)(1 - \rho) + f\psi}.$$
   Under reliable feedback at $f = 0.50$ ($\rho = 0.0438, \psi = 0.0038$), admitted adversarial capacity is $f\psi = 0.0019$ and $f_{\mathrm{eff}} \approx 0.0040$; conversely, when majority feedback channels are compromised ($\rho = 0.0920, \psi = 0.6447$), $f_{\mathrm{eff}} \approx 0.619 > 1/3$, confirming breakdown.
3. *(Formal BFT Safety/Liveness Scope):* Assuming honest-recall $\ge$ malicious-bypass ($1 - \rho \ge \psi$), $f < 1/3$ guarantees $f_{\mathrm{eff}} < 1/3$ strictly. Conversely, when $f \ge 1/3$ globally, no cryptographic or BFT safety guarantee is claimed because $(\rho, \psi)$ depend on measurement integrity (THS, NSI, PRF) which loses majority trust guarantees when $f > 1/3$.

**Proof.**  
By linearity of weight partitioning over disjoint sets $\mathcal{H} \sqcup \mathcal{A}$: $|\mathcal{H} \cap \mathcal{M}_{\mathrm{HM}}|/m = (1-f)(1-\rho)$ and $|\mathcal{A} \cap \mathcal{M}_{\mathrm{HM}}|/m = f\psi$. Forming their ratio yields $f_{\mathrm{eff}}$. Checking $f_{\mathrm{eff}} < 1/3$ gives $3 f\psi < (1-f)(1-\rho) + f\psi \iff 2 f\psi < (1-f)(1-\rho)$. Whenever $f < 1/3$ and $\psi \le 1 - \rho$:
$$2 f\psi < \frac{2}{3}(1 - \rho) < (1 - f)(1 - \rho),$$
since $1 - f > 2/3$. Thus honest-majority non-exclusion preserves the $1/3$ Byzantine margin inside $\mathcal{M}_{\mathrm{HM}}$ whenever $f < 1/3$. Note carefully that if $f = 0.50$ and feedback corruption pushes $\psi > 1/2 \cdot (1-\rho)$, $f_{\mathrm{eff}}$ exceeds $1/3$, violating liveness/safety cleanly as acknowledged in Section 5.1. $\blacksquare$

---

## Theorem 5: Finite-Sample Breakdown Point of Trimmed Dynamic Thresholding & $O(m \cdot n)$ Complexity

**Statement.**  
For $m$ miners and $n=3$ criteria:
1. *(Exact Finite-Sample Breakdown Point):* Let $\alpha_{\mathrm{trim}} \in (0, 0.5)$ denote the symmetric per-tail trimming fraction in Eq. (12). The trimmed mean $\mu_{\mathrm{trim}}$ and trimmed standard deviation $\sigma_{\mathrm{trim}}$ have exact finite-sample breakdown point $\epsilon^* = \lfloor \alpha_{\mathrm{trim}} m \rfloor / m \approx \alpha_{\mathrm{trim}}$ per tail. Thus, with $\alpha_{\mathrm{trim}} = 0.05$, arbitrary outlier injections of size $\le 5\%$ on either tail cause strictly bounded perturbation, whereas larger low-tail noise injections exceeding $5\%$ enter the trimmed interior and shift $\mu_{\mathrm{trim}}$ (as empirically quantified in Section 3.3.4).
2. *(Algorithmic Complexity):* Algorithm 1 requires exact worst-case $O(m \cdot n)$ arithmetic time for TOPSIS scoring and $O(m)$ average time via Quickselect (`np.partition`) for threshold computation (or $O(m \log m)$ via comparison sort), remaining strictly sub-linear per miner.

**Proof.**  
Order closeness coefficients $C_{(1)} \le \dots \le C_{(m)}$ and discard $k = \lfloor \alpha_{\mathrm{trim}} m \rfloor$ elements from each end. Any replacement or corruption of $\le k$ entries at the bottom (or top) stays outside the central window $\{C_{(k+1)}, \dots, C_{(m-k)}\}$, so $\mu_{\mathrm{trim}}$ and $\sigma_{\mathrm{trim}}$ remain strictly bounded inside $[\min_{\text{clean}} C_i, \max_{\text{clean}} C_i]$. When $> k$ extreme values ($0$ or $1$) are injected, at least one injected value enters the central slice, shifting $\mu_{\mathrm{trim}}$ smoothly in proportion to the residual outlier mass. For complexity, column norming, matrix scaling, $A^+/A^-$ scans, and Euclidean distances each traverse $m \times n$ arrays at $O(m \cdot n)$ cost ($n=3$), while Quickselect locates order statistics $k$ and $m-k$ in $O(m)$ expected time. $\blacksquare$
