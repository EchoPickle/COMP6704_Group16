"""Check split integrity, train-only scaling, and reproducible delivery files."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from smo_svm.datasets import make_binary_gaussians

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_experiment.py"


def test_dataset_reproducibility_labels_and_sample_count():
    X, y = make_binary_gaussians(51, 123)
    X2, y2 = make_binary_gaussians(51, 123)
    assert X.shape == (51, 2)
    assert y.shape == (51,)
    assert set(y) == {-1, 1}
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


@pytest.mark.parametrize("count", [0, 3, 4.5, True])
def test_invalid_sample_count(count):
    with pytest.raises(ValueError):
        make_binary_gaussians(count)


def test_split_is_disjoint_complete_stratified_and_reproducible():
    spec = importlib.util.spec_from_file_location("smo_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, y = make_binary_gaussians(300, 42)
    train, test = module.stratified_split(y, 42)
    train2, test2 = module.stratified_split(y, 42)
    assert len(train) == 210
    assert len(test) == 90
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(300))
    assert np.count_nonzero(y[train] == 1) == 105
    assert np.count_nonzero(y[test] == 1) == 45
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)


def test_experiment_cli_outputs_consistent_data_metrics_and_figures(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smo"]["converged"]
    assert metrics["smo"]["max_kkt_violation"] <= metrics["configuration"]["tol"]
    assert metrics["comparison"]["test_prediction_agreement"] >= 0.99
    assert metrics["comparison"]["dual_absolute_difference"] < 1e-5
    with (tmp_path / "data.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300
    train = [r for r in rows if r["split"] == "train"]
    raw_train = np.array([[float(r["x1_raw"]), float(r["x2_raw"])] for r in train])
    scaled_train = np.array([[float(r["x1_scaled"]), float(r["x2_scaled"])] for r in train])
    np.testing.assert_allclose(metrics["data"]["train_mean"], raw_train.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(metrics["data"]["train_scale"], raw_train.std(axis=0), atol=1e-14)
    np.testing.assert_allclose(scaled_train.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(scaled_train.std(axis=0), 1, atol=1e-14)
    saved = np.load(tmp_path / "model.npz")
    assert abs(np.array([int(rows[i]["y"]) for i in saved["train_indices"]]) @ saved["alpha"]) < 1e-9
    for filename in ("experiment.png", "experiment.svg", "experiment.pdf", "history.csv", "report.md"):
        assert (tmp_path / filename).stat().st_size > 100
