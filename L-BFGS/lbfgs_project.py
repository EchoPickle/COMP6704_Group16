#!/usr/bin/env python3
"""Reproducible L-BFGS / squared-hinge SVM project.

Problem/data reference: Cindy0725/COMP6704-Group-24-Convex-Optimization.
The two-loop recursion and optimizer loop are implemented here; the strong-Wolfe
line-search helper is provided by SciPy. See README.md for the experiment protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Set before NumPy import; threadpool_limits also handles already loaded runtimes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import line_search, minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

LABELS = ["Poor", "Standard", "Good"]
NUMERIC = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
           "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
           "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
           "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
           "Amount_invested_monthly", "Monthly_Balance"]
CATEGORICAL = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
SIZES = {"small": 7884, "medium": 13141, "large": 18397}
SOURCES = {
    "reference_project": "https://github.com/Cindy0725/COMP6704-Group-24-Convex-Optimization",
    "dataset": "https://www.kaggle.com/datasets/parisrohan/credit-score-classification",
    "lbfgs": "https://users.iems.northwestern.edu/~nocedal/lbfgs.html",
    "liu_nocedal_1989": "https://doi.org/10.1007/BF01589116",
    "scipy_lbfgsb": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html",
    "scipy_line_search": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html",
}


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False))


def clean_features(df):
    """Deterministic parsing only. Learned imputation/scaling happens after split."""
    X = df[NUMERIC + CATEGORICAL].copy()
    for col in NUMERIC:
        raw = X[col].astype("string").str.strip()
        if col == "Credit_History_Age":
            parts = raw.str.extract(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?")
            X[col] = pd.to_numeric(parts[0], errors="coerce") * 12 + pd.to_numeric(parts[1], errors="coerce")
        else:
            X[col] = pd.to_numeric(raw.str.replace("_", "", regex=False), errors="coerce")
        X[col] = X[col].astype(float)
    for col in CATEGORICAL:
        X[col] = X[col].astype("string").str.strip().replace({"_": pd.NA, "": pd.NA, "!@9#%8": pd.NA})
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def prepare_data(archive, out, seed):
    """Reuse all labelled local data, deduplicate IDs, then split by customer."""
    with zipfile.ZipFile(archive) as z:
        frames = {name: pd.read_csv(z.open(name)) for name in
                  ["train_data_s.csv", "train_data_m.csv", "train_data_l.csv", "test_data.csv"]}
    source_audit = {name: {"rows": len(df), "customers": df.Customer_ID.nunique(),
                          "label_counts": df.Credit_Score.value_counts().to_dict()}
                    for name, df in frames.items()}
    original_overlap = {name: len(set(df.Customer_ID) & set(frames["test_data.csv"].Customer_ID))
                        for name, df in frames.items() if name != "test_data.csv"}
    df = pd.concat(frames.values(), ignore_index=True).drop_duplicates("ID").reset_index(drop=True)
    assert df.Credit_Score.isin(LABELS).all() and df.Customer_ID.notna().all()
    pool_idx, hold_idx = next(GroupShuffleSplit(n_splits=1, train_size=.70, random_state=seed)
                              .split(df, groups=df.Customer_ID))
    val_rel, test_rel = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed + 1)
                            .split(df.iloc[hold_idx], groups=df.iloc[hold_idx].Customer_ID))
    val_idx, test_idx = hold_idx[val_rel], hold_idx[test_rel]
    train_groups, val_groups, test_groups = [set(df.iloc[i].Customer_ID) for i in [pool_idx, val_idx, test_idx]]
    assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
    assert len(pool_idx) >= SIZES["large"]
    # Nested, stratified row subsets within the customer-disjoint training pool.
    large, _ = train_test_split(pool_idx, train_size=SIZES["large"], stratify=df.iloc[pool_idx].Credit_Score, random_state=seed)
    medium, _ = train_test_split(large, train_size=SIZES["medium"], stratify=df.iloc[large].Credit_Score, random_state=seed)
    small, _ = train_test_split(medium, train_size=SIZES["small"], stratify=df.iloc[medium].Credit_Score, random_state=seed)
    idx = {"small": np.sort(small), "medium": np.sort(medium), "large": np.sort(large),
           "validation": np.sort(val_idx), "test": np.sort(test_idx)}
    y = df.Credit_Score.map({v: i for i, v in enumerate(LABELS)}).to_numpy()
    X = clean_features(df)
    # Row positions permit exact reproduction without exporting identifiers.
    np.savez_compressed(out / "split_indices.npz", **idx)
    audit = {"seed": seed, "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
             "original_files": source_audit, "original_train_test_customer_overlap": original_overlap,
             "unique_rows": len(df), "unique_customers": df.Customer_ID.nunique(),
             "split_customer_overlap": 0, "unused_training_pool_rows": len(pool_idx) - len(large),
             "splits": {k: {"rows": len(v), "customers": df.iloc[v].Customer_ID.nunique(),
                             "class_counts": np.bincount(y[v], minlength=3)} for k, v in idx.items()}}
    save_json(out / "data_audit.json", audit)
    datasets = {}
    for size in SIZES:
        pre = preprocessor()
        train = np.ascontiguousarray(pre.fit_transform(X.iloc[idx[size]]), dtype=float)
        datasets[size] = {"X": train, "y": y[idx[size]],
                          "X_val": np.ascontiguousarray(pre.transform(X.iloc[val_idx])), "y_val": y[val_idx],
                          "X_test": np.ascontiguousarray(pre.transform(X.iloc[test_idx])), "y_test": y[test_idx],
                          "features": pre.get_feature_names_out().tolist(), "preprocessor": pre}
    return datasets, audit


class SquaredHinge:
    """F = .5||w||² + .5*reg_b*b² + C*mean(max(0,1-y*f)²)."""
    def __init__(self, X, y, C, reg_b=1e-3):
        self.X, self.y, self.C, self.reg_b = X, y.astype(float), C, reg_b
        self.nfev = 0
        self.cached_x = None
        self.cached_fg = None

    def __call__(self, theta):
        if self.cached_x is not None and np.array_equal(theta, self.cached_x):
            return self.cached_fg
        w, b = theta[:-1], theta[-1]
        r = np.maximum(0., 1. - self.y * (self.X @ w + b))
        coeff = (-2. * self.C / len(self.y)) * self.y * r
        f = .5 * (w @ w) + .5 * self.reg_b * b * b + self.C * (r @ r) / len(self.y)
        g = np.r_[w + self.X.T @ coeff, self.reg_b * b + coeff.sum()]
        self.nfev += 1
        self.cached_x, self.cached_fg = theta.copy(), (float(f), g)
        return self.cached_fg


@dataclass
class Result:
    x: np.ndarray
    f: float
    ginf: float
    nit: int
    nfev: int
    seconds: float
    success: bool
    status: str
    trace: list
    skipped_updates: int = 0
    fallback_steps: int = 0

    def summary(self):
        return {k: v for k, v in self.__dict__.items() if k not in ["x", "trace"]}


def two_loop(g, pairs):
    """Compute H_k g without materializing a dense Hessian approximation."""
    q = g.copy()
    alphas = []
    for s, z, rho in reversed(pairs):
        a = rho * np.dot(s, q)
        alphas.append(a)
        q -= a * z
    gamma = np.dot(pairs[-1][0], pairs[-1][1]) / np.dot(pairs[-1][1], pairs[-1][1]) if pairs else 1.
    r = gamma * q
    for (s, z, rho), a in zip(pairs, reversed(alphas)):
        r += s * (a - rho * np.dot(z, r))
    return r


def optimize(fun, x0, method="lbfgs", m=10, gtol=1e-5, maxiter=1000):
    """L-BFGS or steepest descent with the same strong-Wolfe line search.

    SciPy handles only the one-dimensional line search. On failure, Armijo
    backtracking is attempted and unsuitable curvature pairs are skipped.
    Both methods stop only on gradient tolerance, failure, or iteration cap.
    """
    t0 = time.perf_counter()
    x = x0.copy()
    f, g = fun(x)
    pairs, skipped, fallbacks = [], 0, 0
    trace = [{"iteration": 0, "seconds": time.perf_counter() - t0, "objective": f,
              "gradient_inf": float(np.linalg.norm(g, np.inf)), "evaluations": fun.nfev}]
    status = "iteration_limit"
    nit = 0
    for k in range(maxiter):
        if np.linalg.norm(g, np.inf) <= gtol:
            status = "gradient_tolerance"
            break
        p = -two_loop(g, pairs) if method == "lbfgs" else -g
        if not np.isfinite(p).all() or g @ p >= 0:
            p, pairs = -g, []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = line_search(lambda z: fun(z)[0], lambda z: fun(z)[1], x, p,
                                 gfk=g, old_fval=f, c1=1e-4, c2=.9, maxiter=30)
            alpha = search[0] if search[-1] is not None else None
        if alpha is None:
            fallbacks += 1
            alpha = 1.
            for _ in range(50):
                if fun(x + alpha * p)[0] <= f + 1e-4 * alpha * (g @ p): break
                alpha *= .5
            else:
                status = "line_search_failed"
                break
        xn = x + alpha * p
        fn, gn = fun(xn)
        if not np.isfinite(fn) or not np.isfinite(gn).all():
            status = "nonfinite"
            break
        if method == "lbfgs":
            s, z = xn - x, gn - g
            curvature = float(s @ z)
            if curvature > 1e-10 * np.linalg.norm(s) * np.linalg.norm(z) and curvature > 0:
                pairs.append((s, z, 1. / curvature))
                pairs = pairs[-m:]
            else: skipped += 1
        x, f, g = xn, fn, gn
        nit = k + 1
        trace.append({"iteration": nit, "seconds": time.perf_counter() - t0, "objective": f,
                      "gradient_inf": float(np.linalg.norm(g, np.inf)), "evaluations": fun.nfev})
    success = bool(np.linalg.norm(g, np.inf) <= gtol)
    if success: status = "gradient_tolerance"
    return Result(x, f, float(np.linalg.norm(g, np.inf)), nit, fun.nfev,
                  time.perf_counter() - t0, success, status, trace, skipped, fallbacks)


def scipy_reference(fun, x0, maxiter=3000):
    t0 = time.perf_counter()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": 1e-8, "ftol": 1e-14, "maxls": 50, "maxcor": 10})
    f, g = fun(res.x)
    return Result(res.x, f, float(np.linalg.norm(g, np.inf)), res.nit, fun.nfev,
                  time.perf_counter() - t0, bool(res.success), str(res.message), [])


def fit_ovr(data, C, method="lbfgs", m=10, maxiter=1000):
    results = []
    for c in range(3):
        fun = SquaredHinge(data["X"], np.where(data["y"] == c, 1., -1.), C)
        results.append(optimize(fun, np.zeros(data["X"].shape[1] + 1), method, m, maxiter=maxiter))
    return results


def evaluate(results, X, y):
    scores = np.column_stack([X @ r.x[:-1] + r.x[-1] for r in results])
    prediction = scores.argmax(axis=1)
    return {"accuracy": accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, average="macro", labels=[0, 1, 2]),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()}


def predict_dataframe(model_path, raw_dataframe):
    """Predict labels using a model bundle generated by this project."""
    bundle = joblib.load(model_path)
    X = bundle["preprocessor"].transform(clean_features(raw_dataframe))
    theta = bundle["weights"]
    scores = X @ theta[:, :-1].T + theta[:, -1]
    return np.asarray(bundle["labels"])[scores.argmax(axis=1)]


def validate(out):
    """Meaningful checks: analytic gradient, two-loop algebra, solver accuracy."""
    rng = np.random.default_rng(9)
    X, y = rng.normal(size=(70, 8)), rng.choice([-1., 1.], size=70)
    fun = SquaredHinge(X, y, 2.5)
    errors = []
    for theta in [np.zeros(9), rng.normal(size=9), rng.normal(size=9) * .1]:
        f, g = fun(theta)
        h = 1e-6
        numerical = np.array([(fun(theta + h * np.eye(9)[j])[0] - fun(theta - h * np.eye(9)[j])[0]) / (2*h) for j in range(9)])
        errors.append(float(np.linalg.norm(g - numerical) / max(1., np.linalg.norm(g), np.linalg.norm(numerical))))
    assert max(errors) < 1e-6, errors
    # Exactly at the squared-hinge join the first derivative is still defined.
    join = SquaredHinge(np.zeros((1, 1)), np.ones(1), 1.)
    theta = np.array([0., 1.])
    analytic = join(theta)[1][-1]
    numeric = (join(theta + [0., 1e-6])[0] - join(theta - [0., 1e-6])[0]) / 2e-6
    assert abs(analytic - numeric) < 1e-5
    A = np.diag(np.linspace(1., 8., 9))
    pairs = []
    for _ in range(4):
        s = rng.normal(size=9); z = A @ s
        pairs.append((s, z, 1. / float(s @ z)))
    H = np.eye(9) * (pairs[-1][0] @ pairs[-1][1]) / (pairs[-1][1] @ pairs[-1][1])
    for s, z, rho in pairs:
        V = np.eye(9) - rho * np.outer(s, z)
        H = V @ H @ V.T + rho * np.outer(s, s)
    g = rng.normal(size=9)
    loop_error = float(np.max(np.abs(two_loop(g, pairs) - H @ g)))
    assert loop_error < 1e-10
    custom = optimize(SquaredHinge(X, y, 2.5), np.zeros(9), gtol=1e-7)
    ref = scipy_reference(SquaredHinge(X, y, 2.5), np.zeros(9))
    objective_rel = abs(custom.f - ref.f) / max(1., abs(ref.f))
    assert custom.success and ref.success and objective_rel < 1e-8
    check = {"gradient_relative_errors": errors, "join_derivative_error": abs(analytic-numeric),
             "two_loop_max_error": loop_error, "solver_objective_relative_difference": objective_rel,
             "custom": custom.summary(), "scipy": ref.summary(), "passed": True}
    save_json(out / "validation.json", check)
    print("Validation checks passed", flush=True)
    return check


def experiments(data, out, repeats):
    large = data["large"]
    tuning = []
    for C in [.01, .1, 1., 10., 100., 1000.]:
        fits = fit_ovr(large, C)
        metrics = evaluate(fits, large["X_val"], large["y_val"])
        row = {"C": C, "validation_macro_f1": metrics["macro_f1"],
               "validation_accuracy": metrics["accuracy"],
               "all_converged": all(r.success for r in fits),
               "max_gradient_inf": max(r.ginf for r in fits)}
        tuning.append(row)
        print("Tuning", row, flush=True)
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)
    eligible = [r for r in tuning if r["all_converged"]]
    assert eligible, "No converged candidate: increase maxiter before choosing C"
    chosen = max(eligible, key=lambda r: (r["validation_macro_f1"], -r["C"]))
    C = chosen["C"]
    save_json(out / "selected_config.json", {"C": C, "m": 10, "reg_b": .001, "gtol": 1e-5,
                                              "selection_metric": "validation macro-F1", "grid": tuning})
    print("Locked C =", C, flush=True)
    # Final test evaluation begins only after configuration is locked.
    scale_rows = []
    scale_models = {}
    for size, ds in data.items():
        runs = []
        for rep in range(repeats):
            fit = fit_ovr(ds, C)
            metrics = evaluate(fit, ds["X_test"], ds["y_test"])
            row = {"size": size, "n": len(ds["y"]), "d": ds["X"].shape[1] + 1,
                   "repeat": rep + 1, "seconds": sum(r.seconds for r in fit),
                   "iterations": sum(r.nit for r in fit), "evaluations": sum(r.nfev for r in fit),
                   "all_converged": all(r.success for r in fit), "max_gradient_inf": max(r.ginf for r in fit),
                   "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}
            scale_rows.append(row); runs.append(row)
            if rep == 0:
                scale_models[size] = fit
                save_json(out / f"fit_{size}.json", {"results": [r.summary() for r in fit], "test": metrics,
                                                    "feature_names": ds["features"]})
                np.savez_compressed(out / f"weights_{size}.npz", weights=np.stack([r.x for r in fit]))
                if size == "large":
                    joblib.dump({"preprocessor": ds["preprocessor"], "weights": np.stack([r.x for r in fit]),
                                 "labels": LABELS, "C": C, "reg_b": .001}, out / "model.joblib")
        print("Scale", size, "seconds", np.mean([r["seconds"] for r in runs]), flush=True)
    pd.DataFrame(scale_rows).to_csv(out / "scalability_runs.csv", index=False)
    # Benchmark one identical binary objective for a transparent optimizer comparison.
    y_binary = np.where(large["y"] == 2, 1., -1.)
    zero = np.zeros(large["X"].shape[1] + 1)
    ref = scipy_reference(SquaredHinge(large["X"], y_binary, C), zero)
    assert ref.success, ref.status
    save_json(out / "scipy_reference.json", ref.summary())
    bench_rows, representative = [], {}
    for method, m, name in [("gd", 0, "Gradient descent"), ("lbfgs", 3, "L-BFGS m=3"),
                            ("lbfgs", 10, "L-BFGS m=10"), ("lbfgs", 20, "L-BFGS m=20")]:
        for rep in range(repeats):
            result = optimize(SquaredHinge(large["X"], y_binary, C), zero,
                              method=method, m=m, maxiter=3000)
            row = {"method": name, "m": m, "repeat": rep + 1, **result.summary(),
                   "relative_objective_difference": abs(result.f-ref.f)/max(1., abs(ref.f))}
            bench_rows.append(row)
            if rep == 0:
                representative[name] = result
                pd.DataFrame(result.trace).to_csv(out / f"trace_{method}_{m}.csv", index=False)
            print("Benchmark", name, "rep", rep + 1, "seconds", round(result.seconds, 3),
                  "iterations", result.nit, "status", result.status, flush=True)
    pd.DataFrame(bench_rows).to_csv(out / "optimizer_runs.csv", index=False)
    custom = representative["L-BFGS m=10"]
    # A successful small-problem validation alone is insufficient: check real data.
    real_check = {"custom_success": custom.success, "scipy_success": ref.success,
                  "relative_objective_difference": abs(custom.f-ref.f)/max(1., abs(ref.f)),
                  "prediction_agreement": np.mean((large["X"] @ custom.x[:-1]+custom.x[-1] >= 0) ==
                                                   (large["X"] @ ref.x[:-1]+ref.x[-1] >= 0)),
                  "custom_gradient_inf": custom.ginf, "scipy_gradient_inf": ref.ginf}
    assert real_check["relative_objective_difference"] < 1e-6 and custom.success, real_check
    save_json(out / "real_data_validation.json", real_check)
    scale = pd.DataFrame(scale_rows).groupby("size", sort=False).agg(
        n=("n", "first"), d=("d", "first"), seconds_mean=("seconds", "mean"),
        seconds_std=("seconds", "std"), iterations=("iterations", "first"),
        accuracy=("test_accuracy", "first"), macro_f1=("test_macro_f1", "first"),
        all_converged=("all_converged", "all")).reset_index()
    bench = pd.DataFrame(bench_rows).groupby("method", sort=False).agg(
        m=("m", "first"), seconds_mean=("seconds", "mean"), seconds_std=("seconds", "std"),
        iterations=("nit", "first"), evaluations=("nfev", "first"), final_objective=("f", "first"),
        gradient_inf=("ginf", "first"), success=("success", "all"), status=("status", "first")).reset_index()
    scale.to_csv(out / "scalability_summary.csv", index=False)
    bench.to_csv(out / "optimizer_summary.csv", index=False)
    summary = {"C": C, "repeats": repeats, "scale": scale.fillna(0).to_dict("records"),
               "benchmark": bench.fillna(0).to_dict("records"), "real_data_validation": real_check,
               "tuning": tuning, "dimension": len(zero), "sources": SOURCES}
    save_json(out / "summary.json", summary)
    return summary


def make_plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.bbox": "tight"})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"Gradient descent": "#A66741", "L-BFGS m=3": "#557C93", "L-BFGS m=10": "#007F78", "L-BFGS m=20": "#8465A7"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag, label in [("gd_0", "Gradient descent"), ("lbfgs_10", "L-BFGS m=10")]:
        df = pd.read_csv(out / f"trace_{tag}.csv")
        axes[0].plot(df.seconds, df.objective, label=label, color=colors[label])
        axes[1].semilogy(df.seconds, df.gradient_inf, label=label, color=colors[label])
    axes[0].set_ylabel("Objective F"); axes[1].set_ylabel("Gradient infinity norm")
    axes[1].axhline(1e-5, ls="--", color="gray", lw=1, label="Tolerance")
    for ax in axes: ax.set_xlabel("Elapsed training time (s)"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "convergence.png"); fig.savefig(figures / "convergence.pdf"); plt.close(fig)
    scale = pd.read_csv(out / "scalability_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(scale.n, scale.seconds_mean, yerr=scale.seconds_std, marker="o", color="#007F78", capsize=4)
    axes[0].set_ylabel("OvR fit time (s), mean ± SD")
    axes[1].plot(scale.n, scale.macro_f1, marker="o", label="Macro-F1", color="#007F78")
    axes[1].plot(scale.n, scale.accuracy, marker="s", label="Accuracy", color="#557C93")
    axes[1].set_ylabel("Held-out test score"); axes[1].legend()
    for ax in axes: ax.set_xlabel("Training rows"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "scalability.png"); plt.close(fig)
    bench = pd.read_csv(out / "optimizer_summary.csv")
    mem = bench[bench.m > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(mem.m.astype(str), mem.seconds_mean, yerr=mem.seconds_std, capsize=4, color="#007F78")
    ax.set_xlabel("History size m"); ax.set_ylabel("Binary fit time (s), mean ± SD")
    fig.tight_layout(); fig.savefig(figures / "memory_sensitivity.png"); plt.close(fig)
    metrics = json.loads((out / "fit_large.json").read_text())["test"]
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png"); plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validate", "run", "plots", "all"], default="all")
    parser.add_argument("--data", type=Path, default=base / "data" / "data.zip",
                        help="Reference archive containing the four labeled CSV files; see data/README.md")
    parser.add_argument("--output", type=Path, default=base / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2: parser.error("Use at least 2 repeats for timing standard deviation")
    if args.stage in ["run", "all"] and not args.data.is_file():
        parser.error(f"Data archive not found: {args.data}. Run prepare_data.py, or pass --data /path/to/data.zip.")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage in ["validate", "all"]: validate(args.output)
        if args.stage in ["run", "all"]:
            env = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__, "threads": threadpool_info(),
                   "seed": args.seed, "repeats": args.repeats, "run_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                   "timing_scope": "binary solver / sum of three sequential OvR solvers; excludes preprocessing and prediction"}
            save_json(args.output / "environment.json", env)
            data, audit = prepare_data(args.data, args.output, args.seed)
            print("Data prepared:", {k: v["X"].shape for k,v in data.items()}, flush=True)
            experiments(data, args.output, args.repeats)
        if args.stage in ["plots", "all"]: make_plots(args.output)
    print("Completed:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
