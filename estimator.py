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

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# Fraction of the FLOP budget spent on the sampling pass.
#
# THIS IS NOT A LEVER, and it has now been measured properly rather than
# assumed. Raw MSE falls as 1/N while the multiplier rises as N, so the two
# cancel almost exactly:
#
#     raw(f) = V/N,  N ~ f*B/per_sample     =>  score(f) = C * (1 + r/f)
#     C = V*per_sample/B = 3.64e-7,  r = residual share ~ 0.0015
#
# Five graded submissions on the real grader, 2026-08-08:
#
#   id         f     N        raw         mult    resid    score      V = raw*N
#   #325981  0.60   36,470  6.4190e-7    0.6052  0.0052  3.8846e-7    0.02341
#   #325982  0.45   27,314  8.0295e-7    0.4547  0.0047  3.6512e-7    0.02193
#   #325983  0.30   18,158  1.1380e-6    0.3043  0.0043  3.4624e-7    0.02066
#   #325986  0.15    9,004  2.6024e-6    0.1539  0.0039  4.0049e-7    0.02343
#   #325987  0.11    6,562  3.3023e-6    0.1138  0.0038  3.7564e-7    0.02167
#
# **V = raw*N is constant at 0.02222 +/- 5.4%.** That is the whole story: raw
# MSE is exactly V/N with no floor, so the model above holds and the TRUE score
# spread across f = 0.11..0.60 is only **1.1%** — while the run-to-run noise on
# V is 5%. Every apparent ordering in that score column is noise. A three-point
# fit on f = 0.60/0.45/0.30 looked like a clean 12% gain and a bias floor
# A = 1.45e-7; extending to f = 0.15 and 0.11 destroyed it. **Do not re-derive
# a budget-fraction lever from fewer than ~5 points spanning a wide f range.**
#
# 0.30 is therefore chosen on robustness, not score. Higher f is better by
# ~0.25%, which is not worth having; what 0.30 buys is distance from the two
# cliffs:
#
#   * BUDGET EXHAUSTION zeroes the whole MLP. At f = 0.30 residual would have to
#     reach 1.9 s to trigger it, against 0.014 s measured — a 136x margin
#     (78x at f = 0.60).
#   * The 0.1 MULTIPLIER FLOOR is a cliff, not a slope: below f + r = 0.1 the
#     multiplier stops falling while raw MSE keeps rising, so f = 0.08 is
#     strictly worse than anything above it. 0.30 sits 3x clear.
_BUDGET_FRACTION = 0.30

# Newton-Schulz iterations for the inverse square root (matmul-only whitening).
_NS_ITERS = 12

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
    """Sample-count-independent FLOPs of whitening: the NS loop plus the check."""
    return (3.0 * _NS_ITERS + 4.0) * width ** 3


def _sample_count(mlp: MLP, budget: int, *, whitening: bool) -> int:
    """Largest even sample count fitting the budget fraction.

    Per sample: `2*n*n*L` for the forward pass, plus `2*n*n` for whitening
    (one `X^T X` to form the covariance, one `X @ M` to apply the transform).
    Rounded up generously — over-reserving costs utilisation, not score.
    """
    n, L = mlp.width, mlp.depth
    per_sample = 2.0 * n * n * L + (4.0 * n * n if whitening else 0.0)
    reserved = _whiten_overhead_flops(n) if whitening else 0.0
    k = int((_BUDGET_FRACTION * float(budget) - reserved) / per_sample)
    return max(2, k - (k % 2))  # even, for antithetic pairing


def _antithetic(width: int, n_samples: int, seed: int):
    """Antithetic sample block: rows are x and -x, so odd moments vanish exactly.

    `.astype` already returns a flopscope array, so wrapping it in `fnp.array`
    only bought a second copy — and since flopscope 0.9 a copy is billed at one
    FLOP per element written, so that wrapper cost `width * n_samples / 2`
    for nothing.
    """
    rng = fnp.random.default_rng(seed)
    half = rng.standard_normal((n_samples // 2, width)).astype(fnp.float32)
    return fnp.concatenate([half, -half], axis=0)


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
    C = fnp.matmul(x.T, x) / float(n)
    C = (C + C.T) * 0.5                      # explicit symmetry
    C = C + fnp.eye(width, dtype=fnp.float32) * _JITTER
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
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
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
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
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
