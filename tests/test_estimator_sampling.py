"""Regression coverage for budget-aware whitening selection."""

from types import SimpleNamespace

import flopscope.numpy as fnp

import estimator


def _capture_sampling(monkeypatch, budget):
    calls = {}
    mlp = SimpleNamespace(width=1024, depth=16, seed=42, weights=[fnp.eye(1, dtype=fnp.float32)])

    def half_block(width, n_samples, seed):
        calls.update(width=width, n_samples=n_samples, seed=seed)
        return fnp.zeros((1, 1), dtype=fnp.float32)

    def whitening_matrix(half, width):
        calls["whitening"] = True
        return None  # Exercise the existing unwhitened fallback, too.

    monkeypatch.setattr(estimator, "_half_block", half_block)
    monkeypatch.setattr(estimator, "_whitening_matrix", whitening_matrix)
    monkeypatch.setattr(estimator, "_forward_layer_means", lambda mlp, half, weight: half)
    estimator._monte_carlo(mlp, budget)
    return calls


def test_current_budget_reclaims_rejected_whitening_cost(monkeypatch):
    calls = _capture_sampling(monkeypatch, 2**41)
    assert "whitening" not in calls
    assert calls["n_samples"] == 5130  # Previously 4170, followed by a rejected transform.
    assert calls["seed"] == 42


def test_larger_sample_block_keeps_verified_whitening(monkeypatch):
    calls = _capture_sampling(monkeypatch, 2**43)
    assert calls["whitening"]
    assert calls["n_samples"] // 2 >= 4 * calls["width"]


def test_antithetic_pairs_do_not_count_as_independent_rows(monkeypatch):
    mlp = SimpleNamespace(width=1024, depth=16)
    samples = estimator._sample_count(mlp, 2**41, whitening=True)
    assert samples >= 4 * mlp.width
    assert samples // 2 < 4 * mlp.width
    assert "whitening" not in _capture_sampling(monkeypatch, 2**41)


def test_degenerate_whitening_still_falls_back():
    half = fnp.ones((16, 8), dtype=fnp.float32)
    assert estimator._whitening_matrix(half, 8) is None
