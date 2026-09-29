# Submission improvement — 27 September 2026

The current estimator wasted its whitening reservation at the tuned Phase 2
budget. It planned 4,170 antithetic samples, but those contain only 2,085
independent rows for a 1,024-dimensional covariance. Four Newton–Schulz steps
failed the existing whitening check on seeds 0, 1, and 42: errors ranged from
7.15e-6 to 7.33e-6, against a 1e-8 tolerance. The estimator therefore paid for
the Gram matrix and iterations before using unwhitened samples anyway.

`estimator.py` now requires at least four independent rows per dimension before
attempting whitening. When the planned block is too small, it skips the attempt
and reallocates the reservation to sampling. This gives 5,130 samples (+23.0%)
at the current shape and budget. Larger blocks retain the original whitening
algorithm and numerical guard. Blend weight, calibration scale, RNG seed,
budget fraction, and covariance propagation are unchanged.

## Paired accuracy benchmark

Compared with the source in `submission-reviewed-20260906.tar.gz` (verified
SHA-256 equality), using all 16 networks in the first public mini parquet shard
at revision `v2-phase2`. Targets are the published 1e9-sample estimates.

| Metric | Previous source | Improved source |
|---|---:|---:|
| Mean adjusted final-layer score | 1.3000899843e-7 | 1.2739720882e-7 |
| FLOPs per network | 219,623,086,043 | 218,601,007,068 |
| Budget utilization | 9.9873% | 9.9408% |
| Maximum measured residual time | 0.10340 s | 0.10452 s |
| Failed networks | 0/16 | 0/16 |

Average score improved **2.0089%**, with lower error on **13/16** networks.
Both versions retain the 0.1 score multiplier. This is a paired local subset
result, not a new leaderboard score or a full-public/private-set guarantee.

`scripts/compare_submission.py` downloads only seeds and target columns, then
reconstructs weights using the public protocol-3.0 generation procedure. It
checks analytic first-layer means against the baked first-layer targets before
scoring. It pins BLAS to one thread, alternates baseline/candidate execution
order, meters each prediction separately, and saves per-network results.
This reconstruction and target access exist only in offline evaluation tooling;
the packaged estimator receives only the normal MLP and budget inputs.

Report: `submission-improvement-benchmark.json`. Comments and the module
docstring were clarified during the benchmark; the final executable AST was
verified identical to the evaluated candidate, and both source hashes are
recorded in the report.

Reproduce after extracting the previous archive's `estimator.py` to a separate
baseline path:

```powershell
.venv/Scripts/python.exe scripts/compare_submission.py --baseline path/to/baseline.py --output comparison.json
```

The current official pre-submission checklist was also consulted:
https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/pre-submission-checklist.md

## Validation

- Four sampling/whitening regression tests passed.
- Ruff passed on the changed Python files.
- `whest validate --estimator estimator.py`: every displayed check OK.
- Local and subprocess runners: three generated 1024x16 networks, seed 42,
  1,000 reference samples, one BLAS thread. Exact per-network MSE and FLOP
  parity; zero failures in either runner. These low-sample references check
  runner parity only; the accuracy comparison above uses baked public targets.
- Maximum subprocess residual: 0.07996 s. Reports:
  `submission-improved-local.json` and `submission-improved-subprocess.json`.
- `whest validate-package submission-improved-20260927.tar.gz` passed.
  Archive contains only `estimator.py` and `manifest.json`; packaged source
  matches the working file byte for byte.
- Archive SHA-256:
  `d72a63f33225739ad17d650bf4e7eceda5c59d0813a02cbd2df3c814304f5d4e`.

Windows checks do not establish Linux sandbox memory enforcement or performance
on grader hardware. No upload or leaderboard submission was made in this review.
