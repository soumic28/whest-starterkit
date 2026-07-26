#!/usr/bin/env bash
# WhestBench local validation — run this once to reproduce every check.
# Usage (Git Bash):  bash run_checks.sh
# Does NOT submit. Read STATUS_REPORT.md for what each result means.

set -e
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

DATASET="hf://aicrowd/arc-whestbench-public-2026@v1-phase1"

echo "==================================================================="
echo " 0. Health check"
echo "==================================================================="
uv run whest doctor

echo
echo "==================================================================="
echo " 1. Stage 1 — local math vs Monte Carlo (depth 32)"
echo "==================================================================="
uv run python estimator.py

echo
echo "==================================================================="
echo " 2. Stage 2 — contract validation"
echo "==================================================================="
uv run whest validate --estimator estimator.py

echo
echo "==================================================================="
echo " 3. Stage 3 — local runner, full mini split (100 MLPs)"
echo "==================================================================="
uv run whest run --estimator estimator.py --dataset "$DATASET" --split mini --runner local

echo
echo "==================================================================="
echo " 4. Stage 4 — subprocess runner (grader-like isolation)"
echo "==================================================================="
uv run whest run --estimator estimator.py --dataset "$DATASET" --split mini --runner subprocess

echo
echo "==================================================================="
echo " 5. Parity spot-check — local vs subprocess must agree (seed 42, 3 MLPs)"
echo "==================================================================="
uv run whest run --estimator estimator.py --dataset "$DATASET" --split mini --runner local      --seed 42 --n-mlps 3
uv run whest run --estimator estimator.py --dataset "$DATASET" --split mini --runner subprocess --seed 42 --n-mlps 3

echo
echo "==================================================================="
echo " 6. Package the submission tarball (build only — does NOT submit)"
echo "==================================================================="
uv run whest package --estimator estimator.py --output submission.tar.gz
tar tf submission.tar.gz   # should list estimator.py + manifest.json

echo
echo "All checks done. Nothing was submitted."
