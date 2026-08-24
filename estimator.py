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
# PHASE 2 CHANGED WHAT THIS TRADES OFF, AND IT IS STILL NOT MUCH OF A LEVER.
#
# Phase 1 priced residual wall time into the score at lambda = 1e11, so
# score(f) = C*(1 + r/f) and higher f was very slightly better. Five graded
# submissions spanning f = 0.11..0.60 confirmed the true spread was 1.1%
# against 5% run-to-run noise: raw*N was constant at 0.02222 +/- 5.4%, so raw
# MSE is exactly V/N and the multiplier's rise cancels it.
#
# Phase 2 sets lambda = 0 (C_m = F_m), which removes the residual term
# entirely. What is left is the fixed cost of whitening, which does not scale
# with the sample count:
#
#     score(f) = (V/N) * (F_fixed + N*c_sample) / B
#              = V*c_sample/B  +  V*F_fixed/(N*B)
#
# The first term is a constant floor. Only the second moves, so higher f is
# better — but measured at the Phase 2 shape F_fixed = 4.3e10 against
# N*c_sample = 6.0e11, so it is 6.7% of the bill at f = 0.30 and 3.4% at
# f = 0.58. **Doubling f buys 3.5%.**
#
# THE BINDING CONSTRAINT IS NO LONGER THE BUDGET. Phase 2 replaced the priced
# residual with a hard 400 ms per-MLP cap, and an MLP that crosses it is scored
# against zeros at multiplier 1.0. One such MLP in a suite of 100 contributes
# ~0.9/100 = 9e-3 to a mean that is otherwise ~1e-6, so a single residual
# overrun is roughly four orders of magnitude worse than every optimisation in
# this file put together. Residual is dominated by the RNG draw, which scales
# with N, so raising f spends the safety margin to buy the 3.5%:
#
#     f = 0.30   residual 0.164 s   2.4x margin
#     f = 0.58   residual ~0.21 s   1.9x margin   (+3.5% score)
#
# That is not a trade worth making, and it points the opposite way from the
# Phase 1 conclusion for a reason that has nothing to do with score. f = 0.30
# also stays 3x clear of the 0.1 multiplier floor, which is a cliff rather than
# a slope: below it the multiplier stops falling while raw MSE keeps rising.
_BUDGET_FRACTION = 0.30

# Newton-Schulz iterations for the inverse square root (matmul-only whitening).
#
# Six, not twelve. NS converges quadratically and the sample covariance starts
# close to I, so at the Phase 2 shape the whitening residual measured
# mean((M^T C M - I)^2) = 1.5e-5, 5.5e-7, 1.5e-9, 3.5e-14, 9.2e-16 over the
# first five iterations and then sits on the float32 floor. It is converged by
# 4 even at N = 8,000 (d/N = 0.128), the noisiest block this estimator can
# draw. Each iteration is three width^3 matmuls = 6.4e9 FLOPs at width 1024,
# so the six unused ones were costing 3.9e10 — 5.5% of the whole bill — to
# re-derive a number that had stopped moving. The guard below is what makes
# trimming this safe: if six were ever too few, the check fails and the
# unwhitened block is forwarded.
_NS_ITERS = 6

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


def _whiten_overhead_flops(width: int) -> float:
    """Sample-count-independent FLOPs of whitening: the NS loop plus the check.

    Each NS iteration is three `width x width` matmuls at `2*width^3` each, and
    the check is two more. The old form of this used `3*_NS_ITERS + 4`, which
    dropped the factor of two in `2*n^3` and so under-reserved by half; at
    width 256 that was 0.15% of the budget and invisible, but at width 1024 the
    same error is 1.8% and showed up directly as utilisation overshooting its
    target (0.318 measured against f = 0.30).
    """
    return (6.0 * _NS_ITERS + 4.0) * width ** 3


def _sample_count(mlp: MLP, budget: int, *, whitening: bool) -> int:
    """Largest even sample count fitting the budget fraction.

    Per sample: `2*n*n*L` for the forward pass, plus whitening's two passes
    over the block — the Gram `X^T X`, billed at `n*n` because it is formed by
    a symmetric-tagged einsum (see `_gram`), and `X @ M` at `2*n*n`. Reserved
    as `4*n*n` rather than `3*n*n` so that the plain-matmul fallback inside
    `_gram` cannot push the run over its own reservation.
    """
    n, L = mlp.width, mlp.depth
    per_sample = 2.0 * n * n * L + (4.0 * n * n if whitening else 0.0)
    reserved = _whiten_overhead_flops(n) if whitening else 0.0
    k = int((_BUDGET_FRACTION * float(budget) - reserved) / per_sample)
    return max(2, k - (k % 2))  # even, for antithetic pairing


def _antithetic(width: int, n_samples: int, seed: int):
    """Antithetic sample block: rows are x and -x, so odd moments vanish exactly.

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
    return fnp.concatenate([half, -half], axis=0)


def _gram(x, n_rows: int):
    """`X^T X / n` — via einsum, which flopscope bills at half the matmul rate.

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


def _forward_layer_means(mlp: MLP, x) -> fnp.ndarray:
    """Push samples through every layer, averaging post-ReLU activations.

    The weights arrive as float64 and are cast to float32 first: without the
    cast every `matmul` promotes to float64 and, from flopscope 0.9.0 on, bills
    at 2.0x — which alone puts this pass over budget. The cast itself is
    width^2 per layer, five orders of magnitude below the matmul it protects.
    """
    rows = []
    for w in mlp.weights:
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


def _whitened_block(x, width: int):
    """Return (block, whitened) — the whitened block, or `x` if the check fails.

    Runs BEFORE the forward pass, so a failure here costs only the O(width^3)
    already spent: the caller simply forwards the unwhitened samples and lands
    on plain antithetic MC instead of on an expensive second attempt.
    """
    n = x.shape[0]
    C = _gram(x, n)
    fnp.fill_diagonal(C, fnp.diag(C) + _JITTER)
    try:
        M = _inverse_sqrt(C, width)
        if _whitening_error(M, C, width) <= _WHITEN_TOL:
            return fnp.matmul(x, M), True
    except Exception:
        pass
    return x, False


def _monte_carlo(mlp: MLP, budget: int) -> fnp.ndarray:
    """Antithetic MC, whitened when the whitening transform verifies."""
    width = mlp.width
    whitening = (
        _whiten_overhead_flops(width)
        <= _WHITEN_MAX_OVERHEAD_FRACTION * _BUDGET_FRACTION * float(budget)
    )
    x = _antithetic(width, _sample_count(mlp, budget, whitening=whitening), mlp.seed)
    if whitening:
        x, _ = _whitened_block(x, width)
    # Last line of defence for the dtype rate: a float64 block would double the
    # cost of every layer below. A no-op when the block is already float32.
    return _forward_layer_means(mlp, fnp.asarray(x, dtype=fnp.float32))


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
        mu = mean
        rows.append(mean * math.exp(log_scale) if log_scale != 0.0 else mean)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Verified-whitened antithetic Monte Carlo.

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

        try:
            pred = _monte_carlo(mlp, budget)
        except Exception:
            try:
                pred = _covariance_propagation(mlp)
            except Exception:
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
