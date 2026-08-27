# WhestBench 2026 — Status Report

**Project:** ARC White-Box Estimation Challenge 2026 (`whest-starterkit`)
**Last updated:** 2026-08-26
**Live round:** **Phase 2** (round 1429), 22 Aug → **17 Oct 2026**
**Best submission:** **`#328499`** — 6.4427 × 10⁻⁷ (14.5% better than the port)

> Phase 1 and Phase 2 scores are **not comparable** — different shape, budget and
> suite. 7.07 × 10⁻⁷ here is not a regression from Phase 1's 3.46 × 10⁻⁷.

---

## 0. What Phase 2 changed

| | Phase 1 | **Phase 2** |
|---|---|---|
| shape | 256 × 32 | **1024 × 16** → `predict()` returns (16, 1024) |
| budget `B_m` | 2.72e11 | **2⁴¹ = 2,199,023,255,552** |
| effective compute | `C = F + λR` | **`C = F`** (λ = 0) |
| residual wall time | priced at 1e11 | **hard cap 0.4 s — over it, the MLP is ZEROED** |
| wall per `predict()` | 60 s | 120 s |
| `setup()` | — | 5 s, overrun fails the whole submission |
| memory | 64 GB | **8 GB** |
| submissions/day | 48 | **10** |
| unmetered compute | priced | **prohibited, disqualifiable** |

AIcrowd required **re-accepting the challenge rules** before it would take a
Phase 2 submission — `whest submit` failed until that was done in the browser.

**The grader lags the kit**: it runs `flopscope 0.11.0` / `whestbench 0.15.0`
while the kit pins 0.12.0 / 0.16.0. Verified in a throwaway 0.11.0 venv that both
billing optimisations below hold under it.

---

## 1. Results

| id | change | N | raw MSE | mult | score |
|---|---|--:|--:|--:|--:|
| #328224 | Phase 2 port of the Phase 1 method | 16,338 | 2.5770e-6 | 0.2924 | 7.5354e-7 |
| #328226 | + whitening folded into W₁ | 17,238 | 2.4496e-6 | 0.2920 | 7.1525e-7 |
| #328227 | ablation: whitening **off** | 19,660 | 3.3764e-6 | 0.3002 | 1.0137e-6 |
| #328228 | + f = 0.50 on the old structure | 29,756 | 1.4000e-6 | 0.4862 | 6.8067e-7 |
| #328352 | + NS = 5, f = 0.30 | 17,420 | 2.4216e-6 | 0.2919 | 7.0692e-7 |
| #328489 | + half-block Gram, NS = 4 | 18,134 | 2.3540e-6 | 0.2959 | 6.9656e-7 |
| #328497 | + layer 1 on the half block | 18,700 | 2.2787e-6 | 0.2958 | 6.7399e-7 |
| **#328499** | **+ f = 0.50 — BEST** | 31,808 | 1.3073e-6 | 0.4928 | **6.4427e-7** |

**14.5% better than the port.** `estimator.py` on branch `phase2` is #328499.

### The cost model is exactly predictive

The grader's multiplier matched locally-computed `flops_used / 2⁴¹` to **five
significant figures on every single submission**. With `V = raw × N` stable at
0.0416–0.0427 (±1%),

```
score(f) = V·c_sample/B  +  V·F_fixed/(N·B)
```

predicts a change before it is submitted — the fold was predicted at 5.3% and
graded 5.1%. The multiplier match also **proves 0/100 failures**, since any
failed MLP forces the multiplier to 1.0.

## 2. What actually helped

**a. Fold the whitening into the first layer — 5.1%, the biggest win.**
Whitening then running layer 1 is `(X @ M) @ W₁`; by associativity that equals
`X @ (M @ W₁)`, so the `X @ M` pass over the whole sample block **disappears**.
It cost `2·N·width²` = 3.4e10 FLOPs and is replaced by a single `2·width³` =
2.1e9 product. Verified exact before submitting: relative RMS difference between
the two orderings was 2.3e-7, about 4 float32 eps. It also cut residual from
0.164 s to 0.097 s, which is what made everything after it possible.

**This was worth only 0.6% at Phase 1's width 256** — `width³` is 64× smaller
relative to `N·width²` when the width drops 4×. It went unnoticed for all of
Phase 1 because at that shape it genuinely did not matter. *Re-check every cost
tradeoff when the shape changes; the negligible ones can become the largest.*

**b. `einsum("ni,nj->ij", x, x)` bills exactly half of `matmul(x.T, x)`** —
17,149,939,200 against 34,266,415,104. flopscope infers symmetry from the
repeated operand and bills only the upper triangle. The discount is priced into
**einsum only**: `as_symmetric` tags on the Newton–Schulz matmuls measured
*more* expensive (51,711,580,136 vs 51,535,419,392), even though every NS
iterate is symmetric.

**c. `standard_normal(dtype=float32)`** — bills 16/element instead of float64's
32 and removes the `.astype` copy. The FLOPs are noise; the **78 ms off the
residual** is the point.

**d. Layer 1 only needs the half block — ~3%, and it lowers the floor.**
A linear map commutes with negation, so for `X = [H; −H]`

```
[H; −H] @ W₁  ==  [H@W₁ ; −(H@W₁)]  ==  [Z; −Z]
```

One matmul over `N/2` rows plus a negation replaces a matmul over `N` rows:
39,245,465,600 → 19,646,668,800, a 49.9% cut on that layer. It stops at layer 1
because ReLU is not odd, so the halves genuinely diverge afterwards. Unlike the
other items this **lowers `c`** (32.5n² → 31.5n²) and so lowers the floor itself
rather than merely approaching it.

Checked and rejected as an extension: at layer 2, `ReLU(−Z) = ReLU(Z) − Z` needs
two `N/2`-row matmuls — exactly the cost of one `N`-row matmul. No saving past
layer 1.

**e. The same identity gives the Gram for free.** `XᵀX = 2·HᵀH`, so the
covariance comes from half the rows (18.28e9 → 9.14e9). Between this and (d) the
full block is never materialised at all. Both exact to float32 rounding.

**f. Newton–Schulz needs 4 iterations, not 12** — measured whitening residual
`mean((MᵀCM − I)²)` = 1.5e-5, 5.5e-7, 1.5e-9, 3.5e-14, then flat on the float32
floor. The guard reports 5.5e-15 against a 1e-8 tolerance. Stopped at 4 rather
than 3: 3 passes (1.5e-9) but with only ~7× margin, and a guard failure costs
the entire 1.57× whitening gain.

**g. Fixed a real bug:** `_whiten_overhead_flops` dropped the factor of 2 in
`2·n³`, under-reserving by half. Invisible at width 256 (0.15% of budget), 1.8%
at width 1024 — it showed as utilisation overshooting 0.30 to 0.318.

---

## 3. *** THE RESIDUAL CAP, AND A MEASUREMENT MISTAKE WORTH KEEPING ***

Phase 2 hard-caps residual wall time at 400 ms; an MLP over it is scored against
zeros **at multiplier 1.0**. One such MLP in 100 adds ~9e-3 to a mean that is
otherwise ~6.5e-7, so a single overrun is ~5 orders of magnitude worse than every
optimisation here combined. That asymmetry governs `_BUDGET_FRACTION`.

**The mistake.** f was dropped from 0.50 to 0.30 after repeated runs showed
f = 0.50 producing erratic residuals (0.213…0.450 s) and one MLP over the cap.
Those measurements were **garbage**. The machine had run its disk down to
0.24 GB, which caps the Windows page file; allocation stalls were landing in the
residual bucket, and runs eventually died on 3.91 MiB allocations. After freeing
27 GB, the same builds measure:

```
f = 0.30   0.087 0.088 0.091 0.099 0.104 0.107 0.113    3.5× margin
f = 0.50   0.135 0.137 0.138 0.140 0.141 0.142 0.143    2.8× margin   <- chosen
f = 0.70   0.184 0.189 0.190 0.191 0.193 0.194 0.198    2.0× margin
f = 0.85   0.230 0.231 0.232 0.234 0.235 0.235 0.236    1.7× margin
```

tight and stable, 0 failures throughout. The control that settles it: the
f = 0.50 build that had **already graded 0/100 on the real grader** measures
0.131…0.137 s on the healthy box — indistinguishable.

**Two rules.** (1) When one failure costs ~5 orders of magnitude more than the
prize, judge on the *distribution* of the margin, not one sample — that part was
right, and it is why f stops at 0.50 rather than 0.85. (2) **Check the measuring
instrument before believing an alarming distribution.** The tell was that the
*spread* was wide, not that the mean was high; a sick machine looks exactly like
that, a real margin problem does not.

**Where f stops:** 0.70 buys 0.9% over 0.50 and 0.85 buys 1.2%, while the margin
halves. At 0.85 the gain is 7.7e-9 against 9.1e-3 per zeroed MLP — **1,180,000×**
— and 1.7× margin means a grader only 1.7× slower than this box starts failing.
The f = 0.70/0.85 residuals are perfectly tight, so that is a margin judgement,
not an instrument problem.

---

## 4. Where the headroom is

Whitening is worth keeping: ablation #328227 measured **V 0.06638 → 0.04223, a
1.572× cut**, against a 1.141× break-even (without it you get 14.1% more
samples). Dropping it costs 42% of score. Notably that is *weaker* than Phase 1's
1.76× at width 256 despite `d/N` being 9× larger — the opposite of the prediction.

**The method is now at 1.032× its own floor.** With `c = 2n²L − n²/2 =
33,030,144` billed FLOPs per sample and `V = 0.0416`, the asymptote as f → 1 with
zero fixed cost is `V·c/B` = **6.25 × 10⁻⁷**.

> **The cost side is finished.** What remains is ~3%, all of it in `f`, and it is
> not worth the residual margin. Going materially lower requires reducing **V**,
> the per-sample variance.

Antithetic pairing forces every odd moment to exactly zero; whitening forces the
sample covariance to exactly I. **Degrees 1–3 are exactly handled, and degree 4
is the first untried term** — that is the open direction, and it is the contest's
own research problem. The control-variate family stays closed for a
shape-independent reason: antithetic already makes `mean(xᵢ) = 0` exactly, so any
linear control variate's correction term is identically zero.

Also checked and found negligible: the network is **positively homogeneous**
(ReLU, no biases), so `f(x) = ‖x‖·g(x/‖x‖)` with `‖x‖ ⊥ u`, suggesting
`E[f] = E[R]·E[g(u)]` with `E[R]` known in closed form. The variance ratio is
`(κ − r)/(1 − r)` with `κ = E[R²]/E[R]² = 1 + 1/(2d)`; at d = 1024 that is
**~1.0007**. Radial concentration in high dimension kills it.

---

## 5. Environment health

- **Disk was the hidden failure.** It reached 0.24 GB free, which capped the page
  file and corrupted every residual measurement (§3) before killing runs outright.
  Fixed by deleting the uv cache (regenerable) — now **27 GB free**. Watch this.
- **The dataset's non-streaming path needs ~13 GB** to materialise an arrow cache
  and fails with a misleading `DatasetGenerationError` wrapping
  `OSError: [Errno 28]`. **Use `--streaming`**, which writes nothing.
- **Take residual readings on an idle machine, repeatedly, and read the spread.**

---

## 6. Open items

- [ ] **Nominate `#328499`** (6.4427e-7) if Phase 2 uses Phase 1's nomination
      mechanism. Watch for an organizer email — Phase 1's default was a trap.
- [ ] Degree-4 moment matching — the only identified route to a lower `V`.
- [ ] Regenerate the AIcrowd API key (pasted in chat/shell history during Phase 1).
- [ ] `estimator.py` / `STATUS_REPORT.md` are modified but **uncommitted** on
      branch `phase2`.
