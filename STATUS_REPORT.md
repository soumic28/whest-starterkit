# WhestBench 2026 — Status Report

**Project:** ARC White-Box Estimation Challenge 2026 (`whest-starterkit`)
**Last updated:** 2026-08-08
**Designated submission:** `#325983` — whitened antithetic Monte Carlo, strict float32, f = 0.30

> **Phase 1 was extended to 10 August 2026, 23:59 UTC.** The write-up is due
> 17 August. See §0 for the rule changes that came with the extension — one of
> them can silently cost us the prize ranking.

---

## 0. What changed on 3 August (forum topic 18125)

**Each team nominates up to 2 submissions** for the Phase 1 private
re-evaluation, which is what decides prize rankings — the public board does not.
**If you nominate nothing, AIcrowd takes your top 2 public submissions**, which
for us would have included **#318705**, broken under the current cost model
(~1.1 × 10⁻⁴). This must be overridden explicitly; the organizers said they
will email nominating instructions.

**Participant code now runs on one physical core** (2 vCPUs, down from 16); the
flopscope backend keeps seven. Residual wall time is still charged at
λ = 1e11 FLOP/s, so 0.1 s costs 3.7% of budget. Verified below.

**The cost model was repriced** (flopscope 0.10.0 / whestbench 0.14.0): dtype
rates as in §1a, and data movement is no longer free — copy / fill /
`concatenate` / `astype` / `reshape` / `stack` / `eye` / `ones` bill 1 per
element written, gather / sort bill 4. Still free: `zeros`, `empty`, views,
`diag`, non-copying `asarray`. **float16 gets no discount** (rate 1.0, same as
float32), so there is no dtype lever below float32.

Two exploits were closed deliberately (float64 "packing", residual wall-time
arbitrage). Compliance validation is **manual**, and submissions found to have
circumvented FLOP accounting are disqualified before prize rollover.

---

## 1. Where we stand

| Metric | Value |
|---|--:|
| **Graded score (#325983)** | **3.46 × 10⁻⁷** |
| Graded raw final-layer MSE | 1.14 × 10⁻⁶ |
| Graded compute multiplier | 0.304 |
| vs Monte-Carlo reference (7.0 × 10⁻⁷) | **~2.0×** |
| Failed MLPs | **0** |
| Wall-clock per MLP, one core | 2.4 s (limit 60 s) |
| Residual wall time, grader | 0.0045 of budget |

Public rank before this session was **#169 of 200+** at 3.895 × 10⁻⁷; 3.46 × 10⁻⁷
lands around **#140**.

### Progression

| Submission | Method | Graded | vs sampling |
|---|---|--:|--:|
| #318691 | covariance propagation | 6.62 × 10⁻⁶ | 0.098× |
| #318705 | antithetic Monte Carlo | 6.25 × 10⁻⁷ | 1.12× |
| #318789 / #318802 / #322538 | + whitening, float64 hot path | 0.70 / 6.6e-5 / 6.6e-5 | **failed** |
| #322542 | + whitening, strict float32 | 3.89 × 10⁻⁷ | ~1.8× |
| #325981 | + copy/stats billing fixes, f = 0.60 | 3.88 × 10⁻⁷ | ~1.8× |
| #325982 | f = 0.45 | 3.65 × 10⁻⁷ | ~1.9× |
| **#325983** | **f = 0.30** | **3.46 × 10⁻⁷** | **~2.0×** |

Roughly a **19× improvement** over the first submission.

#325981's raw MSE is **bit-identical** to #322542's (6.419045888605978 × 10⁻⁷),
which confirms both that the estimator's numerics are unchanged and that neither
the 0.10.0 repricing nor the single-core container materially raised our
residual.

### 1b. The budget fraction is a lever after all — worth 1.12×

This report previously called it a non-lever, reasoning that raw MSE falls as
1/N while the multiplier rises as N, so the two cancel. **That is wrong.** Raw
MSE does not go to zero with N; it has a floor:

```
raw(f)   = A + K/f                         A = 1.45e-7, K = 2.98e-7
score(f) = raw(f) · max(0.1, f + r)        r = 0.0045
```

| f | raw | multiplier | score |
|---|--:|--:|--:|
| 0.60 | 6.4190 × 10⁻⁷ | 0.605 | 3.8846 × 10⁻⁷ |
| 0.45 | 8.0295 × 10⁻⁷ | 0.455 | 3.6512 × 10⁻⁷ |
| **0.30** | 1.1380 × 10⁻⁶ | 0.304 | **3.4624 × 10⁻⁷** |

The fit predicted 3.431 × 10⁻⁷ for f = 0.30 against 3.462 × 10⁻⁷ measured, so it
is good to ~1%. Because A > 0 the score has a genuine interior minimum near
**f ≈ 0.10–0.12 (~3.27 × 10⁻⁷)**.

**We shipped 0.30, not the optimum**, taking 12% of the available 19%:

- The 0.1 multiplier floor is a **cliff, not a slope**. Below `f + r = 0.1` the
  multiplier stops falling while raw MSE keeps climbing — f = 0.08 scores
  3.78 × 10⁻⁷, *worse than f = 0.60*. At 0.30 we sit 3× clear of it.
- Low f amplifies residual wall time, charged at λ regardless of N. If R triples
  on the private hardware, f = 0.30 still beats f = 0.12.

A is most likely the **grader's own ground-truth sampling noise** — a property
of the evaluation suite, not of this estimator — so it may move on a differently
sized private suite. The asymmetry still favours the lower fraction: if A holds,
0.30 wins 12%; if A vanished entirely, 0.30 loses only ~1% to 0.60.

**This is invisible locally.** It only shows up in paired submissions on the real
grader, because A belongs to the grader's ground truth.

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

- [ ] **NOMINATE 2 SUBMISSIONS before 10 Aug 23:59 UTC.** Recommended:
      **#325983** (f = 0.30, 3.46 × 10⁻⁷) and **#325981** (f = 0.60,
      3.88 × 10⁻⁷) — the second is the same estimator at double the budget-cliff
      margin, so the pair hedges the one assumption #325983 leans on. Watch for
      the organizers' email. **Do not let the default stand**: it would pick
      #318705, which is broken under the current cost model.
- [ ] **Algorithmic Contribution write-up due 17 Aug 2026** — §1b and §3 are the
      substance; a $500–5,000 community-contribution prize also exists for
      cost-model feedback.
- [ ] **Regenerate the AIcrowd API key** — it was pasted in chat and shell history.
- [x] Dev-only deps removed — `uv sync` against the new pins dropped torch,
      einops, jaxtyping, sympy, networkx and mpmath.
- [ ] Optional: probe **f = 0.15–0.25**, worth a further 3–5% by the fitted model
      and untested. Each probe is one submission against a 48/day quota.

## 8. If work resumes

Our *estimator* is exhausted as a sampler. The remaining route to the leaders is
a **method that survives depth 32** — precisely the contest's open problem. The
Hermite spectrum and the 4.16 × 10⁻⁸ target in §5 define exactly what it must do.

Ranks 5–20 currently show raw MSE ~2 × 10⁻⁷ against our 1.14 × 10⁻⁶ at f = 0.30
(6.4 × 10⁻⁷ at f = 0.60), so their variance-per-FLOP is genuinely ~4× better.
Ranks 1–4 sit at 4 × 10⁻¹⁰, far below the 4.16 × 10⁻⁸ oracle bound this project
measured — which is itself evidence they are not doing honest estimation, and
they are now under manual compliance review.

**The methodological lesson from §1b generalises:** a quantity that is a property
of the *grader* rather than of the estimator cannot be measured locally at all.
Before assuming any score model, spend two paired submissions on it — that is
what turned a documented "non-lever" into a 12% gain in one afternoon.
