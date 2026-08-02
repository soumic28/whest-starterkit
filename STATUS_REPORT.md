# WhestBench 2026 — Status Report

**Project:** ARC White-Box Estimation Challenge 2026 (`whest-starterkit`)
**Last updated:** 2026-08-02
**Designated submission:** `#322542` — whitened antithetic Monte Carlo, strict float32

---

## 1. Where we stand

| Metric | Value |
|---|--:|
| **Graded score (#322542)** | **3.89 × 10⁻⁷** |
| Graded raw final-layer MSE | 6.42 × 10⁻⁷ |
| Graded compute multiplier | 0.607 |
| Adjusted score (mini split, 100 MLPs) | 3.98 × 10⁻⁷ |
| vs Monte-Carlo reference (7.0 × 10⁻⁷) | **~1.8×** |
| Failed MLPs | **0 / 100** |
| Wall-clock per MLP | ~0.9 s (limit 60 s) |

### Progression

| Submission | Method | Graded | vs sampling |
|---|---|--:|--:|
| #318691 | covariance propagation | 6.62 × 10⁻⁶ | 0.098× |
| #318705 | antithetic Monte Carlo | 6.25 × 10⁻⁷ | 1.12× |
| #318789 / #318802 / #322538 | + whitening, float64 hot path | 0.70 / 6.6e-5 / 6.6e-5 | **failed** |
| **#322542** | **+ whitening, strict float32** | **3.89 × 10⁻⁷** | **~1.8×** |

Roughly a **17× improvement** over the first submission.

> **#318705 must not be designated for Phase 2.** Measured under `flopscope==0.10.0`
> — what the grader runs today — it exhausts the budget on 3/3 real MLPs and scores
> ~1.1 × 10⁻⁴. It worked in July only because the grader was on the older flopscope.
> See §1a.

### 1a. The dtype-billing trap (cause of three failed submissions)

From **flopscope 0.9.0** onward, FLOPs are billed at a per-dtype **rate**:
float16/float32 = 1.0, **float64 = 2.0** (`flopscope/_weights.py`,
`_ACTIVE_DTYPE_RATES`). The starter kit pins `flopscope>=0.8.0rc5,<0.9.0`, which
has no such rate — its docs still say *"dtype matters for precision, not FLOPs."*
The grader is on the newer one (whestbench 0.14.0 requires flopscope ≥ 0.10.0).

Two sources of float64 each double the entire forward pass on their own:

- **`mlp.weights` are float64**, so `matmul(x_f32, w_f64)` promotes.
- **`fnp.eye` / `fnp.zeros` default to float64**, so a whitening matrix built
  from them promotes the sample block at `x @ M`.

Submission #322538 validated at 0.68 utilization locally and hit **1.08 on the
grader**, exhausting the budget partway through the forward pass and scoring the
covariance-propagation fallback. The fix is to cast weights with
`fnp.asarray(w, dtype=fnp.float32)` and to pass `dtype=fnp.float32` to every
`eye`/`zeros`. That alone took the score from 6.58 × 10⁻⁵ to 3.89 × 10⁻⁷.

This also **retracts** the earlier conclusion that `fnp.linalg` is not
grader-safe. Running the grader's real `flopscope-client`/`flopscope-server`
stack locally gives bit-identical values to in-process flopscope — including
`cholesky`, `inv`, `.T` and `float(...)` — on both synthetic and real dataset
MLPs. Whitening was never the problem.

**Reproduce the grader before every submission:** install `flopscope==0.10.0`
in a scratch venv, run the estimator on real dataset MLPs whose weights were
saved *without* casting, and check `ctx.flops_used` against the budget. Read
graded diagnostics from `GET https://www.aicrowd.com/api/v1/submissions/{id}`
with `Authorization: Token <key>` — `score_secondary` is the raw MSE, so
`score / score_secondary` is the compute multiplier and immediately reveals a
blown budget.

---

## 2. The method

```
1. Draw N/2 Gaussian samples; append their negatives  (antithetic)
      -> every ODD sample moment is exactly zero
2. Whiten: C = XᵀX/N ;  M = C^(-1/2) by Newton-Schulz (matmul only)
      -> sample covariance is exactly I  (kills the degree-2 error)
3. VERIFY MᵀCM == I. If it fails, forward the unwhitened block instead.
4. Forward all samples in float32; average post-ReLU activations per layer
5. Fallback: covariance propagation (~0.6% of budget) -> zeros
6. Sanitize: finite, shape (depth, width), float32
```

N is sized to 60% of the FLOP budget including whitening's `4·N·n²` overhead.
**Every array on the hot path stays float32** — see §1a.

Two design rules, each bought with a failed submission:

- **The guard checks the property, not a proxy.** `MᵀCM == I` is exactly what
  whitening claims, and costs `2·width³` (~0.01% of budget) because it works on
  the 256×256 covariance rather than the sample block. #318802's
  `mean(x²) ≈ 1` check only caught scale errors.
- **Never chain two expensive methods.** FLOPs spent by a failed attempt are
  never refunded, so "whitening at 60% → plain MC at 60%" needs 120% of budget
  and is guaranteed to exhaust it — that is what #318802 did. Here the guard
  fires *before* the forward pass, so its failure path is free: the same
  forward pass on the same samples, minus the whitening.

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
