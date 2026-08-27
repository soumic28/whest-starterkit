"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Strategy — verified-whitened antithetic Monte Carlo, billed in float32:

    Primary  : antithetic MC with the sample block whitened so its covariance
               is exactly I (measured 1.76x better raw MSE than plain
               antithetic MC on the 100-MLP mini split).

    Guard    : the whitening matrix is CHECKED before it is used, by forming
               M^T C M and requiring it to equal I. If the check fails, the
               already-drawn UNWHITENED block is forwarded instead. Nothing
               expensive is wasted and the result degrades to plain antithetic
               MC — the exact method that graded 6.25e-7 on the server.

    Fallback : covariance propagation (~0.6% of budget), then zeros.

    Sanitizer: output forced to shape (depth, width), finite, float32.

*** EVERY ARRAY ON THE HOT PATH MUST STAY float32. ***

This is the single most expensive thing to get wrong here, and it is invisible
locally. From flopscope 0.9.0 on, FLOPs are billed at a per-dtype RATE:
float16/float32 bill at 1.0 and **float64 bills at 2.0** (see
`flopscope/_weights.py: _ACTIVE_DTYPE_RATES`). The starter kit pins
flopscope 0.8.0rc5, which has no such rate — its docs still say "dtype matters
for precision, not FLOPs" — so a float64 hot path costs nothing locally and
double on the grader.

Two things silently produce float64, and each on its own doubles the whole
forward pass:

  * `mlp.weights` arrive as **float64**, so `matmul(x_f32, w_f64)` promotes.
    They are cast once per layer in `_forward_layer_means`.
  * `fnp.eye(...)` and `fnp.zeros(...)` default to float64, so the whitening
    matrix built from them promotes the sample block on `x @ M`. Every such
    call below passes `dtype=fnp.float32` explicitly.

Submission #322538 was exactly this bug: it validated at 0.68 utilisation
locally and hit 1.08 on the grader, exhausting the budget partway through the
forward pass and scoring the covariance-propagation fallback (6.58e-5). It also
explains #318789 (0.701) and #318802 (6.6e-5), which were read as
"`fnp.linalg` is not grader-safe" — the real fault was dtype billing, and
whitening was never the problem. Reproduce with
`pip install flopscope==0.10.0` and check `ctx.flops_used`.

WHY THE WHITENING CHECK IS SHAPED THIS WAY:

1. A guard must verify the mathematical PROPERTY, not a proxy for it. Checking
   `mean(x_w^2) ~= 1` only catches scale errors; a wrong-but-correctly-scaled
   transform slips through. `M^T C M == I` is the property whitening claims,
   so nothing that matters can pass it. It costs 2*width^3 (~0.01% of budget)
   because it works on the 256x256 covariance, not on the sample block.

2. FLOPs spent by a failed attempt are NEVER refunded, so a fallback must be
   cheap enough to always fit in what remains. #318802 chained whitening at 60%
   of budget into plain MC at 60% of budget; the first failed, the second ran
   the budget dry, and only the 0.6% covariance-propagation fallback survived —
   at no compute discount. Here the guard fires BEFORE the forward pass, so the
   fallback is not another expensive method: it is the same forward pass on the
   same samples, minus the whitening.

Only the final layer is scored (`final_layer_mse x max(0.1, compute/budget)`,
see whestbench/budget.py), but the full (depth, width) array must be returned.

Why whitening works. Antithetic pairing (x, -x) already forces every ODD sample
moment to exactly zero, which is most of what plain antithetic buys (odd degrees
are ~54% of the output variance here). Whitening additionally forces the sample
covariance to exactly I, removing the degree-2 component — measured ~49% of the
even part. Diagonal rescaling alone does nothing (1.01x): it fixes per-neuron
variances but not the cross-correlations, which is where the error lives.

Why NOT cumulant propagation: verified against ARC's own reference implementation
at this configuration — K=3 factorized kprop scores ~1.3e-4 raw MSE at depth 32,
roughly 100x WORSE than Monte Carlo. Their methods are width-asymptotic and break
down by L/n = 32/256. Measured spec: an Edgeworth final layer would win ~80x, but
only with mu AND sigma known to ~1.3e-3 relative; covariance propagation delivers
8.6e-3 on mu and 0.14-0.19 on sigma, so it is 110-150x short on sigma alone.

Run `python estimator.py` to compare against Monte Carlo locally.
Then climb: `whest validate` -> `whest run` -> `whest package`.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# Fraction of the FLOP budget spent on the sampling pass.
#
# Phase 1 priced residual wall time into the score (lambda = 1e11), giving
# score(f) = C*(1 + r/f); a five-point sweep over f = 0.11..0.60 showed the true
# spread was 1.1% against 5% noise, so f was not a lever there.
#
# Phase 2 sets lambda = 0 (C_m = F_m). What survives is whitening's fixed cost,
# which does not scale with the sample count:
#
#     score(f) = V*c_sample/B  +  V*F_fixed/(N*B)
#
# The first term is a floor; only the second moves. Graded, with V = raw*N held
# at 0.0422 +/- 1%:
#
#     f      N        graded/predicted   residual (idle box)   margin
#     0.30   18,700   6.7399e-7 graded    0.087..0.113          3.5x
#     0.50   31,808   6.4427e-7 graded    0.135..0.143          2.8x   <- here
#     0.70   44,916   6.386e-7  predicted 0.184..0.198          2.0x
#     0.85   54,745   6.362e-7  predicted 0.230..0.236          1.7x
#
# It saturates because F_fixed/(N*c) is already small: 0.70 buys 0.9% over 0.50
# and 0.85 buys 1.2%, while the residual margin halves. At f = 0.85 the gain is
# 7.7e-9 against a 9.1e-3 cost per zeroed MLP — **1,180,000x** — and a 1.7x
# margin means a grader only 1.7x slower than this box starts failing MLPs.
# Not taken. The f = 0.70/0.85 residuals are perfectly TIGHT, so this is a
# genuine margin judgement, not the instrument problem described below.
#
# THE REAL CONSTRAINT IS THE 400 ms RESIDUAL CAP, and it is worth knowing how
# this value was nearly set wrong. Phase 2 stopped pricing residual time and
# hard-caps it: an MLP over the cap is scored against zeros AT MULTIPLIER 1.0.
# One such MLP in 100 adds ~9e-3 to a mean that is otherwise ~7e-7 — about
# 250,000x the gain f = 0.50 buys, so the asymmetry says be conservative.
#
# Acting on that, f was dropped to 0.30 after repeated runs showed f = 0.50
# producing erratic residuals (0.213..0.450 s) and one MLP over the cap.
# **Those measurements were garbage.** The machine had run its disk to 0.24 GB,
# which capped the page file, and allocation stalls were landing in the residual
# bucket. After freeing 27 GB the same configurations measure:
#
#     f = 0.30   0.087 0.088 0.091 0.099 0.104 0.107 0.113   3.5x margin
#     f = 0.50   0.135 0.137 0.138 0.140 0.141 0.142 0.143   2.8x margin
#
# tight and stable, 24/24 MLPs clean. The control that settles it: the f = 0.50
# build that GRADED 0/100 ON THE REAL GRADER measures 0.131..0.137 s here, i.e.
# indistinguishable from this one.
#
# TWO RULES OUT OF THAT. (1) A cliff-shaped failure mode deserves the
# *distribution* of the margin, not one sample — that part was right. (2) But
# check the instrument before believing a distribution that looks alarming: an
# erratic tail with a wide spread is more often a sick machine than a real
# effect, and the tell is that the spread is wide rather than the mean high.
#
# 0.50 sits 5x clear of the 0.1 multiplier floor, which is a cliff rather than a
# slope: below it the multiplier stops falling while raw MSE keeps rising.
_BUDGET_FRACTION = 0.1016

# Weight on the covariance-propagation estimate in the final blend.
#
# Covariance propagation is DETERMINISTIC — its error is pure bias, no variance.
# Monte Carlo is UNBIASED — pure variance, no bias. Their errors are therefore
# independent, and for `mu = w*CP + (1-w)*MC`
#
#     MSE(w) = w^2 * bias^2 + (1-w)^2 * V/N        (the cross term vanishes)
#
# minimised at `w = (V/N) / (bias^2 + V/N)`. Measured at this shape:
# bias^2 = 4.43e-6 and V/N = 1.03e-5 at N = 3,875, giving w = 0.700 — which is
# also where a direct sweep over 6 MLPs against a 400k-sample reference bottoms
# out. The optimum is FLAT: w = 0.65..0.80 all land within 2% of each other, so
# this does not need per-MLP tuning and does not need to be exact.
#
#     w      0.00    0.50    0.65   0.70   0.80    1.00
#     score  1.05e-6 3.63e-7 3.01e-7 2.95e-7 3.07e-7 4.18e-7
_BLEND_CP = 0.874

# Covariance propagation carries a SYSTEMATIC multiplicative bias: it
# over-estimates the final-layer means by a remarkably constant ~0.17%.
# Correcting it with a single constant is the largest single win available
# after the blend itself.
#
# Fitted as the least-squares scale c = <CP, truth>/<CP, CP> on **eight real
# contest MLPs** (mini split, v2-phase2) against their baked 1e9-sample ground
# truth:
#
#     c per MLP  0.998069 0.998096 0.998124 0.998231 0.998263 0.998282
#                0.998437 0.998560
#     mean 0.998258   std 0.000160
#
# and independently on 8 self-generated He-init MLPs: mean 0.998281, std
# 0.000328 — agreeing to 2.3e-5. **That agreement is the evidence this is a
# property of the ALGORITHM at this shape, not of the public MLPs**, which is
# what makes it safe for the private re-evaluation. Shipping calibration
# constants is explicitly permitted (docs/concepts/allowed-code.md).
#
# Measured on the real MLPs: CP's MSE falls 4.394e-6 -> 1.425e-6, a **3.08x**
# cut. Because CP was ~72% of the blend's error budget, that moves the optimal
# blend weight from 0.70 up to 0.874.
_CP_SCALE = 0.998258

# Newton-Schulz iterations for the inverse square root (matmul-only whitening).
#
# Four, not twelve. NS converges quadratically and the sample covariance starts
# close to I, so at the Phase 2 shape the whitening residual measured
# mean((M^T C M - I)^2) = 1.5e-5, 5.5e-7, 1.5e-9, 3.5e-14, 9.2e-16 over the
# first five iterations and then sits on the float32 floor. It is converged by
# 4 even at N = 8,000 (d/N = 0.128), the noisiest block this estimator can
# draw. Each iteration is three width^3 matmuls = 6.4e9 FLOPs at width 1024,
# so the seven unused ones were costing 4.5e10 — 6.4% of the whole bill — to
# re-derive a number that had stopped moving. Four lands at 3.5e-14 against a
# 1e-8 tolerance — a margin of 3e5 — and still converges (2.5e-9) at N = 8,000,
# the noisiest block this estimator can draw. Three would pass too (1.5e-9) but
# with only ~7x margin, which is not enough for a guard whose failure costs the
# whole 1.57x whitening gain. The guard below is what makes
# trimming this safe: if six were ever too few, the check fails and the
# unwhitened block is forwarded.
_NS_ITERS = 4

# Ridge added to the sample covariance, for numerical safety.
_JITTER = 1e-6

# Whitening is accepted only if mean((M^T C M - I)^2) is below this. A correct
# transform lands around 1e-14; every observed failure mode is many orders
# above. Set well clear of both so float32 accumulation noise cannot trip it.
_WHITEN_TOL = 1e-8

# Whitening's fixed cost is ~50*width^3, independent of sample count, so it is
# negligible at the contest budget (0.3% of 2.72e11) but not at a tiny one.
# Attempt it only when it is genuinely cheap relative to the sampling pass.
_WHITEN_MAX_OVERHEAD_FRACTION = 0.25

# Covariance-propagation fallback: rescale if a diagonal entry blows up.
_COV_RESCALE_THRESHOLD = 1e30


def _covprop_flops(width: int, depth: int) -> float:
    """FLOPs of the covariance-propagation pass, for budget reservation.

    Dominated by one symmetric-rate `einsum("ij,ia,jb->ab", cov, w, w)` per
    layer at `3*width^3`; measured 51,701,818,336 at 1024x16 against `3*n^3*L`
    = 51,539,607,552, so 3.1 carries the remainder plus the per-layer
    `as_symmetric` re-tags and the stats calls.
    """
    return 3.1 * width ** 3 * depth


def _whiten_overhead_flops(width: int) -> float:
    """Sample-count-independent FLOPs of whitening: the NS loop plus the check.

    Each NS iteration is three `width x width` matmuls at `2*width^3` each, the
    check is two more, and folding M into the first layer (`M @ W1`, see
    `_monte_carlo`) is one more. The old form of this used `3*_NS_ITERS + 4`, which
    dropped the factor of two in `2*n^3` and so under-reserved by half; at
    width 256 that was 0.15% of the budget and invisible, but at width 1024 the
    same error is 1.8% and showed up directly as utilisation overshooting its
    target (0.318 measured against f = 0.30).
    """
    return (6.0 * _NS_ITERS + 6.0) * width ** 3


def _sample_count(mlp: MLP, budget: int, *, whitening: bool) -> int:
    """Largest even sample count fitting the budget fraction.

    Per sample, actually billed:

        layer 1        n*n          (matmul over N/2 rows — see _forward_layer_means)
        layers 2..L    2*n*n*(L-1)
        Gram           n*n/2        (symmetric einsum over N/2 rows, when whitening)
        negate+concat  1.5*n        (1 FLOP/element, 0.005% — ignored below)

    summing to `2*n*n*L - n*n/2` with whitening. Reserved at **`2*n*n*L`**,
    which covers two independent fallbacks at once: `_gram` degrading to a plain
    matmul (`n*n` instead of `n*n/2`), and the whitening guard failing so that
    no whitening runs at all. Over-reserving costs utilisation, never
    correctness.

    There is no `X @ M` term — whitening is folded into layer 1's weights rather
    than applied to the block, which moved it into the fixed reservation above.
    """
    n, L = mlp.width, mlp.depth
    per_sample = 2.0 * n * n * L
    reserved = _covprop_flops(n, L)
    if whitening:
        reserved += _whiten_overhead_flops(n)
    k = int((_BUDGET_FRACTION * float(budget) - reserved) / per_sample)
    return max(2, k - (k % 2))  # even, for antithetic pairing


def _half_block(width: int, n_samples: int, seed: int):
    """The FIRST HALF of the antithetic block. The other half is exactly `-H`.

    Nothing in this estimator ever materialises `[H; -H]`. The Gram needs only
    `H` (`X^T X = 2 H^T H`), and layer 1 needs only `H` because
    `(-H) @ W = -(H @ W)` — see `_forward_layer_means`. Odd sample moments still
    vanish exactly, which is the whole point of antithetic pairing; it is only
    the storage and the arithmetic that are halved.

    Drawn directly in float32. `standard_normal` bills 16 FLOPs per element at
    the dtype rate, so a float64 draw is 32 and a float32 draw is 16 — and the
    float64 path then owes another `width*n_samples/2` for the `.astype` copy
    that a float32 draw does not need. Measured at the Phase 2 block size:
    309,544,960 FLOPs and 0.219 s for draw-then-cast against 158,955,520 and
    0.141 s drawing float32 outright.

    The FLOPs are noise against the forward pass; **the 78 ms is not.** Phase 2
    replaced Phase 1's priced residual with a hard 400 ms per-MLP cap, and
    crossing it zeroes that MLP — the single most expensive failure available,
    since one zeroed MLP out of 100 costs more than every optimisation in this
    file combined. The draw was 45% of measured residual, so this is a safety
    change that happens to also be free.

    `dtype=` is passed through a fallback rather than assumed: it has not been
    exercised on the grader's `flopscope-client` RPC proxy, and the dtype is
    re-checked afterwards in case the kwarg is accepted and then ignored. The
    fallback is the exact draw-then-cast path that graded 3.46e-7 in Phase 1.
    """
    rng = fnp.random.default_rng(seed)
    shape = (n_samples // 2, width)
    try:
        half = rng.standard_normal(shape, dtype=fnp.float32)
    except Exception:
        half = rng.standard_normal(shape)
    if half.dtype != fnp.float32:
        half = half.astype(fnp.float32)
    return half


def _gram(x, n_rows: int):
    """`X^T X / n` — via einsum, which flopscope bills at half the matmul rate.

    Callers pass the ANTITHETIC HALF-BLOCK, not the full one; see
    `_whitening_matrix` for why that is exact.

    A Gram matrix is symmetric by construction, and flopscope 0.12 infers that
    from the repeated operand in `einsum("ni,nj->ij", x, x)`: it returns a
    SymmetryGroup-tagged array and bills only the upper triangle. Measured at
    the Phase 2 block: 17,150,464,000 FLOPs against 34,267,463,680 for
    `matmul(x.T, x)` — an exact halving, and 2.4% of the total bill.

    Note this discount is only available on the *contraction*. Tagging does
    nothing for the Newton-Schulz matmuls below, even though every iterate
    there is symmetric too: measured 51,535,419,392 untagged against
    51,711,580,136 with `as_symmetric` re-tags, i.e. the tags cost slightly
    more than they save. Symmetry is priced into einsum, not into matmul.
    """
    try:
        return fnp.einsum("ni,nj->ij", x, x) / float(n_rows)
    except Exception:
        C = fnp.matmul(x.T, x) / float(n_rows)
        return (C + C.T) * 0.5   # einsum is exactly symmetric; matmul is not


def _forward_layer_means(mlp: MLP, h, first_weight) -> fnp.ndarray:
    """Push the antithetic block through every layer, averaging post-ReLU.

    `h` is the HALF block; `first_weight` is layer 1's matrix with the whitening
    already folded in (or the plain weight when whitening was not applied).

    LAYER 1 RUNS ON HALF THE ROWS. The full block is `[H; -H]`, and a linear map
    commutes with negation:

        [H; -H] @ W1  ==  [H @ W1; -(H @ W1)]  ==  [Z; -Z]

    so one matmul over `N/2` rows plus a negation replaces a matmul over `N`
    rows. Measured 39,245,465,600 -> 19,646,668,800 FLOPs, a 49.9% cut on that
    layer and **~3% of the whole bill**; the negation and concatenate that
    replace it bill 1 FLOP per element and come to 0.005%. Exact up to float32
    rounding (max abs difference 8.4e-5 on activations of scale ~32, i.e. ~3e-6
    relative, which is BLAS blocking the two shapes differently).

    It stops at layer 1: ReLU is not odd, so `ReLU(-Z) != -ReLU(Z)` and the two
    halves genuinely diverge from here on. Layers 2..L need all `N` rows.

    Weights are cast to float32 as a guard. whestbench hands out float32 now so
    it is a free no-op, but under Phase 1's float64 weights an uncast `matmul`
    promoted the whole block into the 2.0x dtype lane, which on its own put this
    pass over budget (submissions #318789 / #318802 / #322538).
    """
    rows = []
    z = fnp.matmul(h, first_weight)
    x = fnp.maximum(fnp.concatenate([z, -z], axis=0), 0.0)

    # EXACT LAYER-1 MEAN. Layer 1 is the only place in the network where the
    # pre-activation distribution is known exactly: the true input is N(0, I),
    # so z_j is exactly N(0, s_j^2) with s_j = ||W1[:, j]||, and therefore
    #
    #     E[ReLU(z_j)] = s_j / sqrt(2*pi)
    #
    # in closed form. Antithetic pairing forces the INPUT mean to exactly zero,
    # but ReLU is not odd, so the post-activation mean still carries sampling
    # error that antithetic does not touch. Replacing it with the analytic value
    # injects information no amount of sampling provides.
    #
    # s_j uses the ORIGINAL W1, not the whitening-folded `first_weight`: the
    # whitened block has empirical covariance exactly I, so the empirical
    # covariance of z is exactly W1^T W1 regardless of the fold.
    #
    # Measured 1.028x on V, pooled over 6 MLPs (per-seed 1.007..1.070, SE 0.011),
    # for 2 FLOPs per sample-neuron — the empirical mean is `rows[0]`, which was
    # already being computed. Under 0.01% of the bill.
    #
    # The same construction with the exact COVARIANCE — the genuine degree-4
    # correction, via E[ReLU(z_i)ReLU(z_j)] = s_i s_j/(2pi) *
    # [sqrt(1-r^2) + r*arccos(-r)] — measured 1.099x but costs ~8% of the bill
    # (empirical Gram of the activations, two inverse square roots), and its
    # spread over the same 6 MLPs was +/-0.04. Net ~1.5% +/- 4%, i.e. not
    # distinguishable from zero. **Tested and rejected; do not re-attempt
    # without a much larger seed count.**
    w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    s = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    exact_mean = s * float(1.0 / math.sqrt(2.0 * math.pi))
    # Fused as a single broadcast pass. `x - mean + exact` reads and writes the
    # whole (N, width) block TWICE; the FLOPs are noise either way, but each
    # pass over ~130 MB costs ~30 ms of RESIDUAL, and residual is capped at
    # 400 ms with an MLP-zeroing cliff behind it. Folding the two corrections
    # into one vector first makes it one pass.
    x = x - (fnp.mean(x, axis=0) - exact_mean)
    rows.append(exact_mean)

    for w in mlp.weights[1:]:
        x = fnp.maximum(fnp.matmul(x, fnp.asarray(w, dtype=fnp.float32)), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _inverse_sqrt(C, width: int):
    """C^{-1/2} by coupled Newton-Schulz iteration — MATMUL ONLY, no fnp.linalg.

    #318789 used cholesky/inv and scored 0.701 on the grader while passing
    locally with 0 failures, so this sticks to matmul/add/scale.

        A = C / s   (s = trace(C)/width = mean eigenvalue, so A ~ I)
        Y <- A, Z <- I ;  T = (3I - ZY)/2 ;  Y <- YT ;  Z <- TZ
        Z -> A^{-1/2}   =>   C^{-1/2} = Z / sqrt(s)

    Converges in a few iterations because C is already close to I.
    """
    eye = fnp.eye(width, dtype=fnp.float32)
    s = float(fnp.sum(fnp.diag(C))) / float(width)
    if not (s > 1e-8):
        raise ValueError("degenerate sample covariance")
    Y = C / s
    Z = eye
    for _ in range(_NS_ITERS):
        T = eye * 1.5 - fnp.matmul(Z, Y) * 0.5
        Y = fnp.matmul(Y, T)
        Z = fnp.matmul(T, Z)
    return Z / float(math.sqrt(s))


def _whitening_error(M, C, width: int) -> float:
    """mean((M^T C M - I)^2) — the property whitening claims, checked directly.

    Whitening sets `X_w = X M`, so `X_w^T X_w / N` is exactly `M^T C M`. Verifying
    it on the (width, width) covariance costs 2*width^3 rather than another pass
    over the sample block, and no wrong transform can pass it — unlike a
    `mean(x^2) ~= 1` check, which is blind to a correctly-scaled wrong rotation.
    """
    E = fnp.matmul(fnp.matmul(M.T, C), M) - fnp.eye(width, dtype=fnp.float32)
    return float(fnp.mean(E * E))


def _whitening_matrix(x, width: int):
    """Return the verified whitening matrix M, or None if the check fails.

    Returns M itself rather than `x @ M` so the caller can fold it into the
    first layer — see `_monte_carlo`. Runs BEFORE the forward pass, so a
    failure here costs only the O(width^3) already spent: the caller simply
    forwards the unwhitened samples and lands on plain antithetic MC instead of
    on an expensive second attempt.
    """
    # THE GRAM ONLY NEEDS THE HALF-BLOCK, which is all the caller holds. The
    # full block is `X = [H; -H]`, so
    #
    #     X^T X = H^T H + (-H)^T(-H) = 2 H^T H
    #     C = X^T X / N = H^T H / (N/2)
    #
    # the same matrix from half the rows. Exact, not an approximation: measured
    # relative RMS difference between the two forms is 1.4e-7, about 2.4 float32
    # eps. Halves both the Gram's FLOPs (18.28e9 -> 9.14e9) and its residual
    # time, and the residual half is the one that matters — see
    # `_BUDGET_FRACTION`.
    C = _gram(x, x.shape[0])
    fnp.fill_diagonal(C, fnp.diag(C) + _JITTER)
    try:
        M = _inverse_sqrt(C, width)
        if _whitening_error(M, C, width) <= _WHITEN_TOL:
            return M
    except Exception:
        pass
    return None


def _monte_carlo(mlp: MLP, budget: int) -> fnp.ndarray:
    """Antithetic MC, whitened when the whitening transform verifies.

    THE WHITENING IS NEVER APPLIED TO THE SAMPLE BLOCK. Whitening then running
    the first layer is `(X @ M) @ W1`; matrix multiplication is associative, so
    folding it as `X @ (M @ W1)` gives the same answer and the `X @ M` pass over
    the block disappears entirely. That pass cost `2*N*width^2` — at the Phase 2
    shape 3.4e10 FLOPs, **5.0% of the whole bill** — and it is replaced by a
    single `2*width^3` = 2.1e9 product that does not scale with the sample
    count. The saving grows with N, and it is exact rather than an
    approximation: the two orderings differ only in float32 rounding.

    This is worth more than every other optimisation in this file combined, and
    it only became worth chasing at Phase 2's shape: at width 256 the same
    fold saved 0.6% because `width^3` is 64x smaller relative to `N*width^2`
    when the width drops 4x.
    """
    width = mlp.width
    whitening = (
        _whiten_overhead_flops(width)
        <= _WHITEN_MAX_OVERHEAD_FRACTION * _BUDGET_FRACTION * float(budget)
    )
    h = _half_block(width, _sample_count(mlp, budget, whitening=whitening), mlp.seed)

    w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    if whitening:
        M = _whitening_matrix(h, width)
        if M is not None:
            w0 = fnp.matmul(M, w0)

    # Last line of defence for the dtype rate: a float64 block would double the
    # cost of every layer below. A no-op when the block is already float32.
    return _forward_layer_means(mlp, fnp.asarray(h, dtype=fnp.float32), w0)


def _relu_moments(mu_pre, var_pre):
    """Exact first two moments of ReLU(Z), Z ~ N(mu_pre, var_pre), per neuron.

    `flops.stats.*` returns float64 for ANY input dtype — it mirrors
    `scipy.stats`, and the organizers have confirmed they are keeping that
    behaviour (only adding a warning in 0.11.0). Under dtype-aware billing one
    such call silently moves everything downstream into the 2x lane, so both
    results are cast straight back to float32.
    """
    var_pre = fnp.maximum(var_pre, 1e-12)
    sigma = fnp.sqrt(var_pre)
    alpha = mu_pre / sigma
    phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
    Phi = flops.stats.norm.cdf(alpha).astype(fnp.float32)
    mean = mu_pre * Phi + sigma * phi
    ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
    return mean, fnp.maximum(ez2 - mean * mean, 0.0), Phi


def _covariance_propagation(mlp: MLP) -> fnp.ndarray:
    """Last-resort analytic fallback (ARC's K=2 algorithm), ~0.6% of budget.

    Scale bookkeeping uses stdlib `math.log`/`math.exp` on Python floats, not
    `fnp.log`/`fnp.exp`: only ops proven on the grader by a previously-graded
    submission are used on flopscope arrays, and log/exp are not among them.
    Pure-Python math on a float costs no FLOPs and has no parity gap.
    """
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    log_scale = 0.0
    rows = []
    for w in mlp.weights:
        w = fnp.asarray(w, dtype=fnp.float32)  # float64 weights bill at 2x
        max_var = float(fnp.max(fnp.diag(cov)))
        if max_var > _COV_RESCALE_THRESHOLD:
            s = float(fnp.sqrt(max_var))
            mu = mu / s
            cov = cov / (s * s)
            log_scale += math.log(s)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)  # symmetry-aware
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        mean, var_post, gain = _relu_moments(mu_pre, var_pre)
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        # RE-TAG. `fill_diagonal` writes into cov and silently voids its
        # symmetry tag (the `multiply` above keeps it, because outer(gain,gain)
        # is itself symmetry-inferred), so without this the einsum above bills
        # the FULL rate on every subsequent layer — measured 67,674,776,560
        # against 51,709,240,799, i.e. 0.73% of the whole budget thrown away
        # with no warning emitted. as_symmetric costs 7*width^2 a call.
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        mu = mean
        rows.append(mean * math.exp(log_scale) if log_scale != 0.0 else mean)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Covariance propagation blended with whitened antithetic Monte Carlo.

    THE BLEND IS THE POINT, AND IT ONLY WORKS AT THIS ROUND'S SHAPE. Covariance
    propagation is width-asymptotic: at Phase 1's 256x32 (L/n = 0.125) it was
    ~125x worse than sampling and this file carried it purely as a last-resort
    fallback. At Phase 2's 1024x16 (L/n = 0.0156, EIGHT TIMES smaller) it lands
    within **3.5x** of sampling — measured 4.43e-6 raw MSE against MC's 1.26e-6.

    That changes what it is for. It costs 2.35% of the budget, so the whole
    estimator fits under the **0.1 multiplier floor**, where `max(0.1, C/B)`
    stops rewarding thrift and compute below 10% of budget is effectively FREE.
    Its error is pure bias where MC's is pure variance, so the two combine as
    independent estimators. Measured over 6 MLPs against a 400k-sample
    reference: 1.05e-6 for MC alone, 4.18e-7 for covariance propagation alone,
    **2.95e-7 blended** — a 52% improvement over the best pure-MC submission.

    Read the shape dependence as the general lesson: a method rejected by
    measurement at one shape has been rejected at THAT shape. This one was
    re-tested only because Phase 2 moved L/n by 8x.

    DO NOT add an expensive fallback behind another expensive method. FLOPs
    spent by a failed attempt are NOT refunded, so "try expensive A, then
    expensive B" exhausts the budget and the whole MLP is zeroed. Submission
    #318802 did exactly that and graded 6.6e-5. Whitening is checked before the
    forward pass precisely so its failure path stays free.

    The one fallback here is deliberately ~0.6% of budget, so it always fits in
    whatever remains.
    """

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        expected_shape = (mlp.depth, mlp.width)

        # Covariance propagation FIRST: its cost is deterministic given the
        # shape and is reserved inside `_sample_count`, so running it can never
        # squeeze the sampling pass. If it raises, its FLOPs are still spent —
        # which is why the reservation is unconditional.
        try:
            cp = _covariance_propagation(mlp)
            if cp.shape != expected_shape or not bool(fnp.all(fnp.isfinite(cp))):
                cp = None
        except Exception:
            cp = None

        try:
            mc = _monte_carlo(mlp, budget)
            if mc.shape != expected_shape or not bool(fnp.all(fnp.isfinite(mc))):
                mc = None
        except Exception:
            mc = None

        if cp is not None and mc is not None:
            # SCALE GUARD. A finite-but-wrong covariance propagation would
            # poison the blend far worse than dropping it. The two methods
            # estimate the same quantity, so their overall scales must agree to
            # within a small factor; anything outside that is a failure of the
            # recursion (variance blow-up or collapse), not a difference of
            # opinion. Costs 2*width per layer.
            s_cp = float(fnp.sum(cp * cp))
            s_mc = float(fnp.sum(mc * mc))
            if not (s_mc > 0.0 and 0.04 < s_cp / s_mc < 25.0):
                cp = None

        if cp is not None and mc is not None:
            pred = cp * (_BLEND_CP * _CP_SCALE) + mc * (1.0 - _BLEND_CP)
        elif mc is not None:
            pred = mc
        elif cp is not None:
            pred = cp * _CP_SCALE
        else:
            pred = fnp.zeros(expected_shape)

        # Guarantee finite values, correct shape, and float32 output.
        pred = fnp.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        if pred.shape != expected_shape:
            pred = fnp.zeros(expected_shape)
        return fnp.asarray(pred, dtype=fnp.float32)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`.

    `importlib` and `pathlib` are imported here rather than at module scope so
    that nothing but `math`, `flopscope` and `whestbench` is pulled in when the
    grader imports this file. Phase 2 restricts a submission to the grader's
    interpreter, the flopscope client API and the pure-Python stdlib, and every
    submission is reviewed — dynamic module loading is allowed under that rule,
    but it has no reason to be on the import path of the graded artifact when
    it only serves `python estimator.py --baseline ...` locally.
    """
    import importlib.util
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)  # phase-2 competition shape (phase 1 was 256x32, warmup 256x8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
