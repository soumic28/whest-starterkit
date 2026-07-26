# WhestBench 2026 — Status Report

**Project:** ARC White-Box Estimation Challenge 2026 (`whest-starterkit`)
**Last updated:** 2026-07-26
**Current submission:** `#318789` — antithetic Monte Carlo + input whitening

---

## 1. Where we stand

| Metric | Value |
|---|--:|
| **Adjusted score (mini split, 100 MLPs)** | **4.12 × 10⁻⁷** |
| Raw final-layer MSE | 6.20 × 10⁻⁷ |
| vs Monte-Carlo reference (7.0 × 10⁻⁷) | **~2×** |
| Failed MLPs | **0 / 100** |
| Compute utilization | 0.665 |
| Wall-clock per MLP | ~1.1 s (limit 60 s) |

### Progression

| Submission | Method | Adjusted | vs sampling |
|---|---|--:|--:|
| #318691 | covariance propagation | 6.62 × 10⁻⁶ | 0.098× |
| #318705 | antithetic Monte Carlo | 6.25 × 10⁻⁷ | 1.12× |
| **#318789** | **+ input whitening** | **~3.5 × 10⁻⁷ (exp.)** | **~2×** |

Roughly a **19× improvement**, with zero failures at every step.

---

## 2. The method

```
1. Draw N/2 Gaussian samples; append their negatives  (antithetic)
      -> every ODD sample moment is exactly zero
2. Whiten: C = XᵀX/N = LLᵀ ;  X_w = X · inv(L)ᵀ
      -> sample covariance is exactly I  (kills the degree-2 error)
3. Forward all samples; average post-ReLU activations per layer
4. Fallbacks: plain antithetic MC -> covariance propagation -> zeros
5. Sanitize: finite, shape (depth, width), float32
```

N is sized to 60% of the FLOP budget including whitening's `4·N·n²` overhead.

---

## 3. Why this is (near) the ceiling for sampling

The error decomposes by Hermite degree. Measured via Mehler's formula
(`Cov(g(x),g(y)) = Σ_d ρ^d Var(g_d)`), at n=256, L=32:

| Degree | Share | Status |
|---|--:|---|
| odd (1,3,5…) | 54% of Var(g) | ✅ killed by antithetic |
| 2 | ~49% of the even part | ✅ killed by whitening |
| **4** | **60–93% of the remainder** | ❌ **out of reach — see below** |
| 6, 8… | remainder | ❌ |

**Degree-4 cannot be matched within budget.** Two independent arguments:

- **Degrees of freedom:** 183,181,376 independent degree-4 constraints vs
  9,086,464 DOF from 35k samples — underdetermined by 20×. No transformation
  of the sample set can fix it.
- **Cubature:** only structured rules can (symmetry satisfies most constraints).
  Budget allows 64,850 samples. The cheapest constructible degree-5 rule
  (Stroud `E_n^{r²}` 5-1, `n²+n+2`) needs **65,794 points = 101.5% of budget** —
  short by 944 points. Stroud 5-2 (`2n²+1`) is 202%. The Möller-type bound
  (33,153 points, 51%) has no known construction in 256 dimensions.

Also ruled out: reducing FLOPs/sample. `2n²L` is irreducible — flopscope charges
by shape (no dtype discount), ReLU sparsity is per-sample so batched gathers
don't help, and low-rank weight approximation destroys correlation.

---

## 4. Experiment ledger — 14 tested, 2 shipped

**Shipped**
- Antithetic sampling — kills all odd-degree error
- Input whitening — **1.76×** on the mini split

**Rejected, with the reason**

| Idea | Result | Why it failed |
|---|--:|---|
| Diagonal Edgeworth correction | ~1.0× | Linear step crushes marginal skew ~100×/layer (CLT) |
| Linear control variate | 1.10× ceiling | ρ_max = 0.30; only 9% of variance is linear in x |
| Truncated-network CV | circular | High ρ (0.93) only at k=31, where E[·] is the original problem |
| Edgeworth on sampled cumulants | 0.98× | Variance-limited: variance 1.2e-6 ≫ bias 4e-8 |
| Rao–Blackwell (Gaussian) | 0.46× | Final pre-activations are far from Gaussian at depth 32 |
| Exact bivariate ReLU covariance | 1.3× | Gain trick was never the bottleneck — the Gaussian *premise* is |
| Hand-derived Hermite κ₃ | corr 0.12–0.51 | Inherited κ₃ from deep layers dominates fresh generation |
| Truncated-depth κ₃ tensor | corr ~0.20 | Built on inaccurate covariance — chicken-and-egg |
| **ARC's own kprop (K=2, K=3)** | **~100× worse than MC** | Verified against their reference code; breaks down at L/n = 1/8 |
| Latin hypercube | 1.01× | The local "2.24×" was seed noise |
| Radial normalization (homogeneity) | 1.03× | Radial part is only ~0.4% of variance |
| Diagonal rescaling | 1.01× | Fixes variances, not cross-correlations |
| Budget fraction tuning | 1.03× max | `adjusted = C(1 + r/f)`; floor is C = 3.72e-7 |
| Degree-5 cubature | infeasible | 101.5% of budget |

---

## 5. Two findings worth keeping

**1. ARC's published method fails at this depth.** Running their *unmodified*
reference implementation at n=256, L=32: K=2 gives 1.38e-4, K=3 factorized gives
1.32e-4 — about **100× worse than plain Monte Carlo**. Our K=2 matches theirs
bit-for-bit, confirming our implementation was always correct. Their results stop
at 12 hidden layers; they name depth scaling as their open problem, and it is
fully realized here.

**2. The target is quantified.** Edgeworth using the *true* κ₃/κ₄ of the final
pre-activations scores **4.16 × 10⁻⁸** — better than rank 1. So a winning method
must compute those cumulants without sampling noise. That is the whole problem,
stated precisely.

---

## 6. Methodology notes (hard-won)

- **Local 4-seed screening is unreliable.** It produced a phantom 2.24× (LHS)
  and understated a real 1.76× (whitening). **Only the 100-MLP mini split decides.**
- **Diagnose before building.** Every build-first attempt was wasted; the two
  diagnostics (shape-vs-moment, Hermite spectrum) each redirected the whole effort.
- Fitting a raw Vandermonde in ρ is ill-conditioned and returns negative
  variances — divide by ρ² first and fit in s = ρ².
- Windows: every `whest` command needs `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.

---

## 7. Open items

- [ ] **Regenerate the AIcrowd API key** — it was pasted in chat and shell history.
- [ ] Remove dev-only deps: `uv pip uninstall torch einops jaxtyping`
      (installed solely to run ARC's reference code).
- [ ] **Designate the final submission before 19 Sep 2026** — that one goes to the
      private re-run and decides prizes. Phase 1 (closes 31 Jul) does not.

## 8. If work resumes

Sampling is exhausted. The only remaining route is a **mechanistic method that
survives depth 32** — which is precisely the contest's open problem, and what the
leaders (2.2–6× better accuracy-per-FLOP than us) evidently found. The Hermite
spectrum and the 4.16e-8 target in §5 define exactly what such a method must do.
