"""Offline paired benchmark against public baked targets, without weight downloads.

Reconstructs public protocol-3.0 weights using the dataset's published input
seeds and whestbench's generator. This is evaluation tooling only; it must never
be imported by the submitted estimator. Checks reconstructed first-layer means
against the baked targets before scoring. Requires the starter kit environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flopscope as flops
import flopscope.numpy as fnp
import fsspec
import numpy as np
import pyarrow.parquet as pq
import requests
from threadpoolctl import threadpool_limits
from whestbench import sample_mlp
from whestbench.seeds import derive_seed_streams


def load_estimator(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("estimator.py"))
    parser.add_argument("--shards", type=int, nargs="+", default=[0])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or any(shard < 0 or shard > 6 for shard in args.shards):
        parser.error("--limit must be positive and --shards must be between 0 and 6")
    base = "https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/"
    response = requests.get(base + "metadata.json", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata["seed_protocol"]["version"] != "3.0":
        raise ValueError("Weight reconstruction requires seed protocol 3.0")
    width, depth = metadata["width"], metadata["depth"]
    budget = 2**41
    estimators = {
        "baseline": load_estimator(args.baseline, "baseline_submission"),
        "candidate": load_estimator(args.candidate, "candidate_submission"),
    }
    report = {
        "dataset": base,
        "ground_truth_samples": metadata["n_samples"],
        "weight_source": "Reconstructed from published protocol-3.0 input seeds",
        "threads": 1,
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]
        },
        "per_mlp": [],
    }
    flops.configure(symmetry_warnings=False)
    with threadpool_limits(limits=1):
        for shard in args.shards:
            url = base + f"data/mini-{shard:05d}-of-00007.parquet"
            with fsspec.open(url, block_size=65536) as stream:
                rows = pq.ParquetFile(stream).read(
                    columns=["mlp_seed", "mlp_name", "final_means", "all_layer_means"]
                ).to_pylist()
            for row in rows:
                if len(report["per_mlp"]) >= args.limit:
                    break
                weight_seed, _, estimator_seed = derive_seed_streams(row["mlp_seed"])
                mlp = sample_mlp(width, depth, fnp.random.default_rng(weight_seed), seed=estimator_seed)
                exact_first = np.sqrt(np.sum(np.asarray(mlp.weights[0]) ** 2, axis=0) / (2 * math.pi))
                first_error = float(np.mean((exact_first - np.asarray(row["all_layer_means"])[0]) ** 2))
                if first_error > 1e-8:
                    raise ValueError(f"Reconstruction sanity check failed: {first_error}")
                result = {"name": row["mlp_name"], "first_layer_sanity_mse": first_error}
                # Alternate execution order to avoid systematically favouring one.
                names = list(estimators)
                if len(report["per_mlp"]) % 2:
                    names.reverse()
                for name in names:
                    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120, quiet=True) as ctx:
                        pred = estimators[name].predict(mlp, budget)
                    pred = np.asarray(pred)
                    if pred.shape != (depth, width) or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid prediction from {name}")
                    mse = float(np.mean((pred[-1].astype(np.float64) - row["final_means"]) ** 2))
                    failed = ctx.residual_wall_time_s > 0.4 or ctx.flops_used > budget
                    score = float(np.mean(np.asarray(row["final_means"]) ** 2)) if failed else mse * max(0.1, ctx.flops_used / budget)
                    result[name] = {
                        "final_layer_mse": mse,
                        "adjusted_final_layer_score": score,
                        "flops_used": ctx.flops_used,
                        "residual_wall_time_s": ctx.residual_wall_time_s,
                        "failed": failed,
                    }
                report["per_mlp"].append(result)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(len(report["per_mlp"]), result["name"],
                      {k: result[k]["adjusted_final_layer_score"] for k in estimators}, flush=True)
    report["summary"] = {
        name: {
            "mean_score": float(np.mean([r[name]["adjusted_final_layer_score"] for r in report["per_mlp"]])),
            "failures": sum(r[name]["failed"] for r in report["per_mlp"]),
            "max_residual_s": max(r[name]["residual_wall_time_s"] for r in report["per_mlp"]),
        }
        for name in estimators
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
