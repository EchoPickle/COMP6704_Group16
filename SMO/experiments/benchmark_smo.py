"""Paired synthetic scaling study and a partner-selection ablation.

All policies share the same SMO update, scan schedule, initialization and KKT
stopping rule. Only the partner ordering changes in RandomPartnerSMO.
No results from external student projects are used.
"""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from smo_svm import ConvergenceWarning, LinearSMO
from smo_svm.datasets import make_binary_gaussians


class RandomPartnerSMO(LinearSMO):
    """Experimental control: uniformly shuffle all eligible partner indices."""

    def _candidate_indices(self, i):
        return self._rng.permutation(np.delete(np.arange(len(self.y_)), i))


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_case(n_train, seed):
    """Independent train/test samples; train-only feature standardization."""
    X, y = make_binary_gaussians(n_train, seed)
    test_X, test_y = make_binary_gaussians(1000, seed + 10000)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / scale, y, (test_X - mean) / scale, test_y, mean, scale


def assert_feasible(alpha, y, C):
    if np.min(alpha) < -1e-10 or np.max(alpha) > C + 1e-10:
        raise AssertionError("dual box constraint violated")
    if abs(y @ alpha) > 1e-8:
        raise AssertionError("dual equality constraint violated")


def run_study(output, sizes, seeds, C=1.0, tol=1e-4, max_iter=5000):
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    rows = []
    # Warm up imports/BLAS without contributing an observation.
    LinearSMO(random_state=0).fit([[-1.0, 0], [1.0, 0]], [-1, 1])
    SVC(kernel="linear").fit([[-1.0, 0], [1.0, 0]], [-1, 1])
    for n_train in sizes:
        for seed in seeds:
            X, y, test_X, test_y, mean, scale = generate_case(n_train, seed)
            np.savez_compressed(output / "cases" / f"n{n_train}_seed{seed}.npz",
                                X_train=X, y_train=y, X_test=test_X, y_test=test_y,
                                train_mean=mean, train_scale=scale)
            reference = SVC(C=C, kernel="linear", tol=1e-7, shrinking=False)
            started = perf_counter()
            reference.fit(X, y)
            reference_time = perf_counter() - started
            ref_alpha = np.zeros(n_train)
            ref_alpha[reference.support_] = np.abs(reference.dual_coef_[0])
            ref_dual = float(ref_alpha.sum() - 0.5 * np.sum(reference.coef_ ** 2))
            ref_scores = reference.decision_function(test_X)
            ref_predictions = reference.predict(test_X)
            rows.append({
                "n_train": n_train, "seed": seed, "method": "SVC", "C": C, "tol": 1e-7,
                "seconds": reference_time, "converged": reference.fit_status_ == 0,
                "scans": None, "pair_updates": None, "max_kkt": None,
                "equality_residual": abs(float(y @ ref_alpha)), "box_violation": 0.0,
                "dual_objective": ref_dual, "duality_gap": None,
                "test_accuracy": float(np.mean(ref_predictions == test_y)),
                "test_macro_f1": float(f1_score(test_y, ref_predictions, average="macro")),
                "prediction_agreement": 1.0, "max_score_difference": 0.0,
                "dual_difference": 0.0, "gram_mib": None,
            })
            # Alternate order across seeds to reduce systematic timing order bias.
            policies = [("Heuristic", LinearSMO), ("Random", RandomPartnerSMO)]
            if seed % 2:
                policies.reverse()
            for name, cls in policies:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model = cls(C=C, tol=tol, max_iter=max_iter, random_state=seed).fit(X, y)
                assert_feasible(model.alpha_, y, C)
                dual_history = np.array([r["dual_objective"] for r in model.history_])
                if np.min(np.diff(dual_history)) < -1e-7:
                    raise AssertionError("dual objective decreased beyond floating-point tolerance")
                predicted = model.predict(test_X)
                row = {
                    "n_train": n_train, "seed": seed, "method": name, "C": C, "tol": tol,
                    "seconds": model.fit_time_, "converged": model.converged_,
                    "scans": model.n_iter_, "pair_updates": model.n_pair_updates_,
                    "max_kkt": model.max_kkt_violation_,
                    "equality_residual": model.equality_residual_,
                    "box_violation": model.history_[-1]["box_violation"],
                    "dual_objective": model.dual_objective_, "duality_gap": model.duality_gap_,
                    "test_accuracy": float(np.mean(predicted == test_y)),
                    "test_macro_f1": float(f1_score(test_y, predicted, average="macro")),
                    "prediction_agreement": float(np.mean(predicted == ref_predictions)),
                    "max_score_difference": float(np.max(np.abs(model.decision_function(test_X) - ref_scores))),
                    "dual_difference": abs(model.dual_objective_ - ref_dual),
                    "gram_mib": model._kernel.nbytes / 2**20,
                }
                rows.append(row)
                write_csv(output / "cases" / f"n{n_train}_seed{seed}_{name.lower()}_history.csv", model.history_)
                print(f"n={n_train} seed={seed} {name}: {model.n_iter_} scans, "
                      f"{model.n_pair_updates_} updates, KKT={model.max_kkt_violation_:.2e}, "
                      f"{model.fit_time_:.4f}s, converged={model.converged_}", flush=True)
                if captured:
                    print("  " + "; ".join(str(w.message) for w in captured), flush=True)
    write_csv(output / "runs.csv", rows)
    summaries = []
    fields = ["seconds", "scans", "pair_updates", "test_accuracy", "test_macro_f1",
              "prediction_agreement", "max_score_difference", "dual_difference", "gram_mib"]
    for n_train in sizes:
        for method in ["Heuristic", "Random", "SVC"]:
            group = [r for r in rows if r["n_train"] == n_train and r["method"] == method]
            summary = {"n_train": n_train, "method": method, "n_runs": len(group),
                       "n_converged": sum(bool(r["converged"]) for r in group)}
            for key in fields:
                values = [r[key] for r in group if r[key] is not None]
                summary[key + "_mean"] = float(np.mean(values)) if values else None
                summary[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(summary)
    write_csv(output / "summary.csv", summaries)
    custom = [r for r in rows if r["method"] != "SVC"]
    paired = []
    for n_train in sizes:
        for seed in seeds:
            h = next(r for r in rows if r["n_train"] == n_train and r["seed"] == seed and r["method"] == "Heuristic")
            r = next(r for r in rows if r["n_train"] == n_train and r["seed"] == seed and r["method"] == "Random")
            paired.append({"n_train": n_train, "seed": seed,
                           "random_over_heuristic_updates": r["pair_updates"] / h["pair_updates"],
                           "random_over_heuristic_seconds": r["seconds"] / h["seconds"],
                           "heuristic_fewer_updates": h["pair_updates"] < r["pair_updates"],
                           "heuristic_faster": h["seconds"] < r["seconds"]})
    write_csv(output / "paired_comparisons.csv", paired)
    metadata = {
        "protocol": {"sizes": sizes, "seeds": seeds, "n_test": 1000, "C": C, "tol": tol,
                     "max_iter": max_iter, "reference_tol": 1e-7, "reference_shrinking": False,
                     "blas_threads": 1, "data": "independent balanced two-class NumPy Gaussians",
                     "timing": "one warmed fit per policy and case; includes Gram and diagnostics; excludes data generation and standardization",
                     "variation": "sample SD across five matched data/optimizer seeds; not a confidence interval"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__,
                        "sklearn": version("scikit-learn")},
        "checks": {"n_custom_runs": len(custom), "n_converged": sum(bool(r["converged"]) for r in custom),
                   "max_kkt": max(r["max_kkt"] for r in custom),
                   "max_equality_residual": max(r["equality_residual"] for r in custom),
                   "max_box_violation": max(r["box_violation"] for r in custom),
                   "min_prediction_agreement": min(r["prediction_agreement"] for r in custom),
                   "max_score_difference": max(r["max_score_difference"] for r in custom),
                   "max_dual_difference": max(r["dual_difference"] for r in custom)},
        "paired": {"heuristic_fewer_updates_cases": sum(p["heuristic_fewer_updates"] for p in paired),
                   "heuristic_faster_cases": sum(p["heuristic_faster"] for p in paired),
                   "total_cases": len(paired)},
        "summaries": summaries,
    }
    (output / "study.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"checks": metadata["checks"], "paired": metadata["paired"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "study")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 300, 1000, 2000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_study(args.output, args.sizes, args.seeds)


if __name__ == "__main__":
    main()
