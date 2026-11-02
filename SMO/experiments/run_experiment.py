"""Reproduce a held-out synthetic experiment and save all source data.

Run from the project root after ``pip install -e '.[experiment]'``.
The solver never imports sklearn; this script uses SVC as an external oracle.
"""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.svm import SVC

from smo_svm import LinearSMO
from smo_svm.datasets import make_binary_gaussians


def stratified_split(y, seed):
    """A NumPy 70/30 split, retaining all rows and both classes in each set."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        count = int(0.7 * len(indices))
        train.extend(indices[:count])
        test.extend(indices[count:])
    return rng.permutation(train), rng.permutation(test)


def save_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def draw_figure(model, reference, X_train, y_train, X_test, y_test, output, alignment_script):
    """Boundary agreement, objective progress, and the actual stopping metric.

    All observations are shown; these are synthetic data. Quantitative curves
    are from one fixed seed, with no implied sampling uncertainty.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    colors = {-1: "#4477AA", 1: "#CC7733"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.33, top=0.79, wspace=0.48)
    for letter, ax in zip("abc", axes):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-13, 13),
                    textcoords="offset points", weight="bold", fontsize=11)
    fig.suptitle("NumPy SMO: linear soft-margin SVM on synthetic data", x=0.065,
                 ha="left", y=0.97, fontsize=13)
    fig.text(0.065, 0.89, f"C = {model.C:g}   |   KKT tolerance = {model.tol:g}   |   "
             f"Train n = {len(y_train)}   |   Test n = {len(y_test)}", fontsize=9, color="#555555")

    boundary = axes[0]
    all_X = np.vstack((X_train, X_test))
    mins, maxs = all_X.min(axis=0) - 0.4, all_X.max(axis=0) + 0.4
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 230), np.linspace(mins[1], maxs[1], 230))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    scores = model.decision_function(grid).reshape(xx.shape)
    baseline_scores = reference.decision_function(grid).reshape(xx.shape)
    boundary.contourf(xx, yy, scores, levels=[-np.inf, 0, np.inf],
                      colors=["#EFF4F8", "#FAF1E8"], alpha=0.8)
    boundary.contour(xx, yy, scores, levels=[-1, 1], colors="#777777", linewidths=0.8, linestyles=":")
    boundary.contour(xx, yy, scores, levels=[0], colors="#202020", linewidths=2.0)
    boundary.contour(xx, yy, baseline_scores, levels=[0], colors="#888888", linewidths=1.5, linestyles="--")
    for label in (-1, 1):
        train_mask, test_mask = y_train == label, y_test == label
        boundary.scatter(*X_train[train_mask].T, s=14, color=colors[label], alpha=0.8,
                         linewidths=0, zorder=3)
        boundary.scatter(*X_test[test_mask].T, s=22, color=colors[label], marker="x",
                         linewidths=0.85, zorder=4)
    boundary.scatter(*model.support_vectors_.T, s=45, facecolors="none",
                     edgecolors="#303030", linewidths=0.6, zorder=5)
    boundary.set(xlabel="Standardized feature 1", ylabel="Standardized feature 2",
                 xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), title="Decision boundary")
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=colors[-1], markersize=4, label="Class -1"),
        Line2D([], [], linestyle="none", marker="o", color=colors[1], markersize=4, label="Class +1"),
        Line2D([], [], linestyle="none", marker="x", color="#444444", markersize=4, label="Test"),
        Line2D([], [], color="#202020", lw=2, label="SMO"),
        Line2D([], [], color="#888888", ls="--", label="SVC"),
        Line2D([], [], color="#777777", ls=":", label="Margin"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color="#303030", markersize=5, label="Support"),
    ]
    boundary.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.27),
                    ncol=3, frameon=False, columnspacing=0.8, handlelength=1.5)

    history = model.history_
    iterations = [r["iteration"] for r in history]
    objective = axes[1]
    objective.plot(iterations, [r["primal_objective"] for r in history],
                   color="#999999", lw=1.25, label="Primal (min)")
    objective.plot(iterations, [r["dual_objective"] for r in history],
                   color="#244C6A", lw=1.8, label="Dual (max)")
    objective.set(xlabel="Scan (including initial state)", ylabel="Objective value", title="Optimization progress")
    objective.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False)

    kkt = axes[2]
    # A plotting floor affects only display, never recorded or checked values.
    violation = np.maximum([r["max_kkt_violation"] for r in history], 1e-16)
    kkt.semilogy(iterations, violation, color="#244C6A", lw=1.8, label="Max KKT violation")
    kkt.axhline(model.tol, color="#999999", lw=1.1, ls="--", label="Requested tolerance")
    kkt.set(xlabel="Scan (including initial state)", ylabel="Absolute margin violation", title="Convergence criterion")
    kkt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False)
    for ax in axes[1:]:
        ax.set_xlim(0, max(1, model.n_iter_))
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    fig.text(0.065, 0.025, "One synthetic split; all samples retained. Standardization uses training statistics only.",
             fontsize=8, color="#555555")

    fig.canvas.draw()
    # Optional authoring QA hook; portable rendering needs no local skill.
    # Pass the alignment helper path to preserve a full rendered-layout audit.
    if alignment_script is not None:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("panel_alignment_qa", alignment_script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.require_matplotlib_panel_alignment(
            fig, json_out=str(output / "experiment.alignment.json"),
            overlay_svg=str(output / "experiment.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    else:
        positions = [ax.get_position().bounds for ax in axes]
        if np.ptp([p[2] for p in positions]) > 1e-10 or np.ptp([p[3] for p in positions]) > 1e-10:
            raise RuntimeError("plot areas must have equal dimensions")
    fig.savefig(output / "experiment.png", dpi=300)
    fig.savefig(output / "experiment.pdf")
    fig.savefig(output / "experiment.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--alignment-script", type=Path, default=None, help="Optional rendered-layout QA helper")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = make_binary_gaussians(300, args.seed)
    train, test = stratified_split(y, args.seed)
    mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    scaled = (X - mean) / scale
    X_train, X_test, y_train, y_test = scaled[train], scaled[test], y[train], y[test]
    model = LinearSMO(C=args.C, tol=args.tol, max_iter=args.max_iter, random_state=args.seed).fit(X_train, y_train)
    if not model.converged_:
        raise RuntimeError("Experiment did not converge; inspect warning or increase --max-iter")
    reference_tol = min(1e-8, args.tol * 0.01)
    started = perf_counter()
    reference = SVC(C=args.C, kernel="linear", tol=reference_tol, shrinking=False).fit(X_train, y_train)
    reference_seconds = perf_counter() - started
    reference_alpha = np.zeros(len(y_train))
    reference_alpha[reference.support_] = np.abs(reference.dual_coef_[0])
    reference_dual = float(reference_alpha.sum() - 0.5 * np.sum(reference.coef_ ** 2))
    reference_scores = reference.decision_function(X_train)
    reference_primal = float(0.5 * np.sum(reference.coef_ ** 2) + args.C * np.maximum(0, 1 - y_train * reference_scores).sum())
    # Compare scores on the whole shown domain as well as observed samples.
    lows, highs = scaled.min(axis=0) - 0.4, scaled.max(axis=0) + 0.4
    gx, gy = np.meshgrid(np.linspace(lows[0], highs[0], 150), np.linspace(lows[1], highs[1], 150))
    query = np.column_stack((gx.ravel(), gy.ravel()))
    metrics = {
        "configuration": {"C": args.C, "tol": args.tol, "max_iter": args.max_iter, "seed": args.seed,
                          "reference_tol": reference_tol, "reference_shrinking": False},
        "data": {"n_train": len(train), "n_test": len(test), "n_features": X.shape[1],
                 "train_mean": mean.tolist(), "train_scale": scale.tolist(), "synthetic": True},
        "smo": {"converged": model.converged_, "stop_reason": model.stop_reason_,
                "scans": model.n_iter_, "pair_updates": model.n_pair_updates_,
                "seconds": model.fit_time_, "support_vectors": len(model.support_),
                "max_kkt_violation": model.max_kkt_violation_, "n_kkt_violations": model.n_kkt_violations_,
                "equality_residual": model.equality_residual_, "box_violation": model.history_[-1]["box_violation"],
                "primal_objective": model.primal_objective_, "dual_objective": model.dual_objective_,
                "duality_gap": model.duality_gap_, "w": model.w_.tolist(), "b": model.bias_,
                "train_accuracy": float(np.mean(model.predict(X_train) == y_train)),
                "test_accuracy": float(np.mean(model.predict(X_test) == y_test))},
        "svc": {"seconds": reference_seconds, "libsvm_iterations": int(reference.n_iter_[0]),
                "support_vectors": len(reference.support_), "dual_objective": reference_dual,
                "primal_objective": reference_primal, "duality_gap": reference_primal - reference_dual,
                "w": reference.coef_[0].tolist(), "b": float(reference.intercept_[0]),
                "train_accuracy": float(np.mean(reference.predict(X_train) == y_train)),
                "test_accuracy": float(np.mean(reference.predict(X_test) == y_test))},
        "comparison": {
            "test_prediction_agreement": float(np.mean(model.predict(X_test) == reference.predict(X_test))),
            "grid_prediction_agreement": float(np.mean(model.predict(query) == reference.predict(query))),
            "max_grid_score_difference": float(np.max(np.abs(model.decision_function(query) - reference.decision_function(query)))),
            "weight_l2_difference": float(np.linalg.norm(model.w_ - reference.coef_[0])),
            "bias_absolute_difference": abs(model.bias_ - float(reference.intercept_[0])),
            "dual_absolute_difference": abs(model.dual_objective_ - reference_dual)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": version("scikit-learn"), "matplotlib": matplotlib.__version__},
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.output / "model.npz", alpha=model.alpha_, w=model.w_, b=model.bias_,
             support=model.support_, train_indices=train, test_indices=test, mean=mean, scale=scale)
    save_csv(args.output / "history.csv", model.history_)
    train_set = set(train.tolist())
    save_csv(args.output / "data.csv", [
        {"sample_id": i, "split": "train" if i in train_set else "test", "y": int(y[i]),
         "x1_raw": X[i, 0], "x2_raw": X[i, 1], "x1_scaled": scaled[i, 0], "x2_scaled": scaled[i, 1]}
        for i in range(len(y))
    ])
    draw_figure(model, reference, X_train, y_train, X_test, y_test, args.output, args.alignment_script)
    a, b, c = metrics["smo"], metrics["svc"], metrics["comparison"]
    report = f"""# NumPy SMO 合成数据实验

使用 NumPy 生成 300 个有重叠的二维高斯双类样本，固定种子 {args.seed}，分层划分训练 {len(train)} / 测试 {len(test)}。所有样本保留；缩放参数只从训练集估计。C={args.C:g}，SMO margin-KKT 容差 {args.tol:g}。参考为相同输入上的 `SVC(kernel='linear', shrinking=False, tol={reference_tol:g})`。

| 指标 | NumPy SMO | SVC |
|---|---:|---:|
| 训练准确率 | {a['train_accuracy']:.4%} | {b['train_accuracy']:.4%} |
| 测试准确率 | {a['test_accuracy']:.4%} | {b['test_accuracy']:.4%} |
| 对偶目标 | {a['dual_objective']:.10f} | {b['dual_objective']:.10f} |
| 原始对偶间隙 | {a['duality_gap']:.3e} | {b['duality_gap']:.3e} |
| 支持向量数 | {a['support_vectors']} | {b['support_vectors']} |
| 训练秒数 | {a['seconds']:.6f} | {b['seconds']:.6f} |

SMO 在 {a['scans']} 次扫描、{a['pair_updates']} 次变量对更新后达到 KKT 容差，最大违例为 {a['max_kkt_violation']:.3e}，超容差样本数为 {a['n_kkt_violations']}。等式约束误差 {a['equality_residual']:.3e}，盒约束违例 {a['box_violation']:.3e}。

测试预测一致率 {c['test_prediction_agreement']:.4%}；图示区域 22,500 个网格点的一致率 {c['grid_prediction_agreement']:.4%}；网格最大分数差 {c['max_grid_score_difference']:.3e}。权重 L2 差 {c['weight_l2_difference']:.3e}，偏置绝对差 {c['bias_absolute_difference']:.3e}，对偶目标绝对差 {c['dual_absolute_difference']:.3e}。

![决策边界与收敛](experiment.png)

左图显示全部训练点、测试叉号及支持向量空心圈；实线是 SMO 边界，灰色虚线是 SVC 边界，点线是 ±1 间隔。中图为每次扫描的原始与对偶目标，右图为最大 KKT 违例及停止阈值。原始目标与 KKT 残差无需单调，对偶目标应在浮点误差范围内非减。没有对曲线进行平滑；绘图对数下限不影响保存的真实残差。

原始数据、分组与缩放数据在 `data.csv`，逐次日志在 `history.csv`，完整指标与版本在 `metrics.json`，alpha、w,b 与划分索引在 `model.npz`。

这是一个种子的教学验证，不是泛化性能研究。训练时间只测一次，包含不同实现的开销，不能据此宣称速度优势；本实现的扫描次数与 LIBSVM 内部迭代也不是同一计量单位。测试集没有参与优化和缩放参数估计。重复数据下 alpha 可能非唯一，因此按目标、决策分数和预测核验。
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
