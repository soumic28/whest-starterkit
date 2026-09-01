# WhestBench 2026 — Status Report

**Project:** ARC White-Box Estimation Challenge 2026 (`whest-starterkit`)
**Last updated:** 2026-08-27
**Live round:** **Phase 2** (round 1429), 22 Aug → **17 Oct 2026**
**Best submission:** **`#328617`** — 1.2799 × 10⁻⁷ (**5.89× better than the port**)

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

| id | change | score | vs port |
|---|---|--:|--:|
| #328224 | Phase 2 port of the Phase 1 method | 7.5354e-7 | — |
| #328499 | cost work: W₁ fold, half-block, NS=4, f=0.50 | 6.4427e-7 | 14.5% |
| #328609 | + exact analytic layer-1 ReLU mean | 6.2025e-7 | 17.7% |
| #328611 | **+ covprop BLENDED with MC at the 0.1 floor** | 3.2767e-7 | 56.5% |
| #328613 | + budget tuned to sit just under the floor | 3.1762e-7 | 57.8% |
| **#328617** | **+ covprop scale calibration (c = 0.998258)** | **1.2799e-7** | **83.0%** |

All graded, all 0/100 failures. `estimator.py` on branch `phase2` is #328617.

**The two structural wins came from re-testing things the record called closed.**
Cost work bought 17.7%; the blend and its calibration bought the other 65%.

### The cost model is exactly predictive

The grader's multiplier matched locally-computed `flops_used / 2⁴¹` to **five
significant figures on every submission**, and the final six sit at exactly
0.10000 — the multiplier floor. The match also **proves 0/100 failures**, since
any failed MLP forces the multiplier to 1.0.

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

## 4. The blend — where the 65% came from

**Covariance propagation reopened at this shape.** The record said it was closed:
~125× worse than Monte Carlo. That was measured at **256×32**. These methods are
width-asymptotic and Phase 2 moved the controlling ratio 8×:

| round | shape | `L/n` | covprop vs MC |
|---|---|--:|--:|
| Phase 1 | 256 × 32 | 0.125 | ~125× worse |
| **Phase 2** | **1024 × 16** | **0.0156** | **3.5× worse** |

3.5×-worse wins because covprop costs only **2.35% of budget**, putting the whole
estimator under the **0.1 multiplier floor** where `max(0.1, C/B)` stops
rewarding thrift and compute below 10% is free. Staying at the floor is optimal:
pushing utilisation to 0.15 or 0.20 makes the score *worse* (3.79e-7, 4.17e-7).

**And the two methods combine.** Covprop is deterministic (pure bias, zero
variance); MC is unbiased (pure variance, zero bias). Errors are independent, so

```
MSE(w) = w²·bias² + (1−w)²·V/N        optimal w = (V/N)/(bias² + V/N)
```

**Covprop's bias is mostly a systematic 0.17% scale error.** One shipped constant
cuts its MSE **3.08×**. Fitted two independent ways:

| source | c | std |
|---|--:|--:|
| 8 **real** contest MLPs vs baked 1e9-sample truth | **0.998258** | 0.000160 |
| 8 self-generated He-init MLPs | 0.998281 | 0.000328 |

**They agree to 2.3e-5** — evidence this is a property of the *algorithm at this
shape*, not something memorised from the public split, which is what makes it
safe for the private re-evaluation. Shipping calibration constants is explicitly
permitted (`docs/concepts/allowed-code.md`). Correcting covprop moved the optimal
weight 0.70 → **0.874**.

**Rejected — self-calibration.** MC is unbiased, so `c` can be estimated per-MLP
as `⟨CP,MC⟩/⟨CP,CP⟩` with no constant and no transfer risk. Measured **worse**
(2.03e-6 vs 1.45e-6): MC noise gives ĉ a std of 0.000627, larger than the true
across-MLP spread of 0.000160.

**Rejected — exact bivariate ReLU covariance** inside covprop (replacing the gain
trick with `E[ReLU(zᵢ)ReLU(zⱼ)]` by 12-node Gauss–Hermite): **0.26×, 3.8× worse**.
ReLU's kink at zero breaks Gauss–Hermite's smoothness assumption.

**Rejected — exact layer-1 covariance / positive homogeneity** (see git history):
1.099× ± 0.04 against an 8% cost, and 1.0007× respectively.

### Where the remaining error is

Error budget is now **87% covprop bias, 13% MC variance**. After calibration
`b² = 1.425e-6`, and the across-MLP spread in `c` explains only ~1.6% of it — the
rest is **shape** error in covprop's per-neuron pattern, not magnitude. So
refining `c`, or predicting it per-MLP from a cheap observable, is capped at
~1.6%. Any further gain has to correct the shape.

A billing trap found here: **`fill_diagonal` silently voids a symmetry tag with
no warning**, so the per-layer `einsum("ij,ia,jb->ab", cov, w, w)` was billing the
full rate — 67,674,776,560 instead of 51,701,818,336, **0.73% of the whole
budget**. Re-tag with `flops.as_symmetric(cov, symmetry=(0, 1))` after any write
into `cov`.

## 5. Pre-submission verification (run 2026-08-27 against `#328617`)

Every item on `docs/how-to/pre-submission-checklist.md`, plus the
`docs/concepts/allowed-code.md` rules.

### Correctness

| check | result |
|---|---|
| `whest validate` — **every row** OK, not just the panel header | 4/4 OK |
| `setup(context)` within the graded cap | 0.00 s (cap **5 s**) |
| `predict()` returned shape | correct, finite |
| local runner, seed 42, 3 MLPs | 1.614103e-7 |
| subprocess runner, seed 42, 3 MLPs (must match within ~1%) | 1.614103e-7 — **bit-identical** |

### Budget hygiene

| check | result | cap |
|---|--:|--:|
| `budget_exhausted` on every MLP | **false** | — |
| `flops_used` | 217,639,587,803 (**0.09897**) | 2,199,023,255,552 |
| multiplier | **0.10000** (the floor) | — |
| `residual_wall_time_s`, multi-thread | 0.057–0.094 s | **0.4 s** |
| `residual_wall_time_s`, **`--max-threads 1`** | 0.078–0.144 s (**2.8×** margin) | 0.4 s |
| `residual_wall_time_exhausted` | **false** | — |
| `time_exhausted` / wall per predict | 5.7 s single-core (**21×**) | 120 s |
| peak process memory | **0.261 GB** (31× margin) | **8 GB** |
| no clock-calibrated internal deadline | none — `grep` for `time.*` is empty | — |

`--max-threads 1` is the meaningful stress test: the grader pins **2 vCPUs** to
the solution. Residual stays at 2.8× margin there.

### Allowed code (`docs/concepts/allowed-code.md`)

| rule | result |
|---|---|
| module-level imports | `math`, `flopscope`, `flopscope.numpy`, `whestbench` — nothing else |
| vendored numpy/scipy/BLAS | none |
| compiled kernels, ctypes/cffi/FFI | none |
| asyncio / threads / subprocess / multiprocessing | none |
| compute while a flopscope op is in flight | none — no callbacks, no lazy objects |
| touching the flopscope client / transport / accounting | none (`flops_used` appears only in comments) |
| meaningful computation in residual time | none — residual is layer loops and control flow |

`argparse`, `importlib.util` and `pathlib` are imported **lazily inside the
`__main__` block only**, so the graded import path pulls in nothing but the four
modules above. All are pure-Python stdlib and none is on the prohibition list.

### Package

`whest validate-package` passes; the shipped `estimator.py` is **sha256-identical**
to the working tree; manifest declares whestbench 0.16.0 / flopscope 0.12.0.

### Shape-agnosticism

Nothing is pinned to 1024×16 — the estimator reads `mlp.width` / `mlp.depth`:

| shape | output | utilisation | |
|---|---|--:|---|
| 1024×16 (graded) | (16, 1024) | 0.0991 | OK |
| 1024×12 | (12, 1024) | 0.0988 | OK |
| 768×16 | (16, 768) | 0.1001 | OK |
| 1024×20 | (20, 1024) | 0.0992 | OK |
| 256×32 (Phase 1) | (32, 256) | 0.1011 | OK |

### One judgement call, stated plainly

`_CP_SCALE = 0.998258` is fitted partly on **public** MLPs. The rules permit
shipping calibration constants, and tuning *to the public split* is pointless
by design because the private re-evaluation uses fresh seeds — so the question
is whether this constant is a property of the MLPs or of the algorithm. Evidence
that it is the algorithm: an independent fit on **self-generated** He-init MLPs
gives 0.998281 against the real MLPs' 0.998258, agreeing to **2.3e-5**, and the
across-MLP standard deviation is only 0.000160. It should therefore transfer to
unseen seeds at the same shape. It is *not* validated for a different shape —
if the round's shape ever moves, refit it.

## 6. Environment health

- **Disk was the hidden failure.** It reached 0.24 GB free, which capped the page
  file and corrupted every residual measurement (§3) before killing runs outright.
  Fixed by deleting the uv cache (regenerable) — now **27 GB free**. Watch this.
- **The dataset's non-streaming path needs ~13 GB** to materialise an arrow cache
  and fails with a misleading `DatasetGenerationError` wrapping
  `OSError: [Errno 28]`. **Use `--streaming`**, which writes nothing.
- **Take residual readings on an idle machine, repeatedly, and read the spread.**

---

## 7. Open items

- [ ] **Nominate `#328617`** (1.2799e-7) if Phase 2 uses Phase 1's nomination
      mechanism. Watch for an organizer email — Phase 1's default was a trap.
- [x] Degree-4 moment matching — tested at layer 1 (§4). Mean correction banked;
      covariance correction rejected as not distinguishable from zero.
- [ ] Regenerate the AIcrowd API key (pasted in chat/shell history during Phase 1).
- [ ] `estimator.py` / `STATUS_REPORT.md` are modified but **uncommitted** on
      branch `phase2`.
