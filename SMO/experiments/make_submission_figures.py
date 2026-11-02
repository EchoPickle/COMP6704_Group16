"""Figures for the independently implemented SMO report and Beamer section.

Figure contract: quantify optimality on one representative synthetic split,
then show measured computational cost and the effect of partner ordering.
The paired ablation uses every matched seed; no favorable cases are selected.
Backend: Python/Matplotlib. PDF/SVG text is editable; PNGs are 300 dpi previews.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-script", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    results = project / "results"
    alignment = None
    if args.alignment_script is not None:
        spec = importlib.util.spec_from_file_location("figure_alignment_qa", args.alignment_script)
        alignment = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = alignment
        spec.loader.exec_module(alignment)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    colors = {"Heuristic": "#A02337", "Random": "#527F91", "SVC": "#777777"}

    def export(fig, name):
        fig.canvas.draw()
        if alignment is not None:
            alignment.require_matplotlib_panel_alignment(
                fig, json_out=str(args.output / (name + ".alignment.json")),
                overlay_svg=str(args.output / (name + ".alignment.svg")),
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
            )
        fig.savefig(args.output / (name + ".pdf"))
        fig.savefig(args.output / (name + ".svg"))
        fig.savefig(args.output / (name + ".png"), dpi=300)
        plt.close(fig)

    def label_panels(axes):
        for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
            ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-13, 9),
                        textcoords="offset points", weight="bold", fontsize=11)

    history = read_csv(results / "history.csv")
    scans = np.array([int(r["iteration"]) for r in history])
    primal = np.array([float(r["primal_objective"]) for r in history])
    dual = np.array([float(r["dual_objective"]) for r in history])
    kkt = np.maximum([float(r["max_kkt_violation"]) for r in history], 1e-16)
    elapsed = np.array([float(r["elapsed_seconds"]) for r in history])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.33, top=0.86, wspace=0.48)
    label_panels(axes)
    axes[0].plot(scans, primal, color="#8A8A8A", label="Primal", lw=1.4)
    axes[0].plot(scans, dual, color=colors["Heuristic"], label="Dual", lw=1.8)
    axes[0].set(xlabel="Scan", ylabel="Objective", title="Objective values", xlim=(0, scans[-1]))
    axes[1].semilogy(scans, kkt, color=colors["Heuristic"], lw=1.8, label="Max KKT violation")
    axes[1].axhline(1e-5, color="#777777", ls="--", lw=1.1, label="Tolerance")
    axes[1].set(xlabel="Scan", ylabel="Margin violation", title="Optimality check", xlim=(0, scans[-1]))
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.37), ncol=2, frameon=False,
                  handlelength=1.5, columnspacing=0.8)
    export(fig, "convergence")

    # Larger, separate axes are provided for slides as well as the report pair.
    fig, ax = plt.subplots(figsize=(5.7, 2.9))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.27, top=0.95)
    ax.semilogy(1000 * elapsed, kkt, color=colors["Heuristic"], lw=2, label="Max KKT violation")
    ax.axhline(1e-5, color="#777777", ls="--", lw=1.1, label="Tolerance")
    ax.set(xlabel="Elapsed training time (ms)", ylabel="Margin-KKT violation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    export(fig, "kkt_time")

    runs = read_csv(results / "study" / "runs.csv")
    pairs = read_csv(results / "study" / "paired_comparisons.csv")
    sizes = sorted({int(r["n_train"]) for r in runs})
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.30, top=0.95)
    for method, marker in [("Heuristic", "o"), ("Random", "s"), ("SVC", "^")]:
        values = [np.array([float(r["seconds"]) for r in runs
                            if int(r["n_train"]) == n and r["method"] == method]) for n in sizes]
        center = [np.median(v) for v in values]
        low, high = [v.min() for v in values], [v.max() for v in values]
        ax.fill_between(sizes, low, high, color=colors[method], alpha=0.10, linewidth=0)
        ax.plot(sizes, center, color=colors[method], marker=marker, lw=1.7,
                markersize=4, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set(xlabel="Training samples", ylabel="Training time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    export(fig, "scaling")

    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.24, top=0.95)
    all_reductions = []
    for x, n in enumerate(sizes):
        reductions = np.array([100 * (1 - 1 / float(r["random_over_heuristic_updates"]))
                               for r in pairs if int(r["n_train"]) == n])
        all_reductions.extend(reductions.tolist())
        # Every point is one matched data/optimizer seed. Horizontal jitter
        # separates the five points only; it does not encode another variable.
        ax.scatter(x + np.linspace(-0.13, 0.13, len(reductions)), reductions,
                   color=colors["Heuristic"], s=28, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [np.median(reductions)] * 2,
                color="#333333", lw=2.0)
    ax.axhline(0, color="#888888", ls="--", lw=0.9)
    ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
    ax.set(xlabel="Training samples", ylabel="Pair updates saved (%)", ylim=(-3, 100))
    export(fig, "ablation")

    # Complementary panels for one presentation frame: measured runtime and
    # the within-case change in accepted updates. All five seeds are retained.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.33, top=0.87, wspace=0.55)
    label_panels(axes)
    for method, marker in [("Heuristic", "o"), ("Random", "s"), ("SVC", "^")]:
        values = [np.array([float(r["seconds"]) for r in runs
                            if int(r["n_train"]) == n and r["method"] == method]) for n in sizes]
        axes[0].fill_between(sizes, [v.min() for v in values], [v.max() for v in values],
                             color=colors[method], alpha=0.10, linewidth=0)
        axes[0].plot(sizes, [np.median(v) for v in values], color=colors[method],
                     marker=marker, lw=1.6, markersize=3.5, label=method)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(sizes, [str(n) for n in sizes])
    axes[0].set(xlabel="Training samples", ylabel="Training time (s)", title="Measured cost")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=3, frameon=False,
                   handlelength=1.2, columnspacing=0.6, fontsize=8)
    for x, n in enumerate(sizes):
        values = np.array([100 * (1 - 1 / float(r["random_over_heuristic_updates"]))
                           for r in pairs if int(r["n_train"]) == n])
        axes[1].scatter(x + np.linspace(-0.13, 0.13, len(values)), values,
                        color=colors["Heuristic"], s=20, zorder=3)
        axes[1].plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="#333333", lw=1.8)
    axes[1].axhline(0, color="#888888", ls="--", lw=0.8)
    axes[1].set_xticks(range(len(sizes)), [str(n) for n in sizes])
    axes[1].set(xlabel="Training samples", ylabel="Pair updates saved (%)", ylim=(-3, 100),
                 title="Partner-selection ablation")
    export(fig, "efficiency")

    # Reconstruct the representative model from saved numerical data.
    observations = read_csv(results / "data.csv")
    model_data = np.load(results / "model.npz")
    metrics = json.loads((results / "metrics.json").read_text())
    standardized = np.array([[float(r["x1_scaled"]), float(r["x2_scaled"])] for r in observations])
    labels = np.array([int(r["y"]) for r in observations])
    is_test = np.array([r["split"] == "test" for r in observations])
    low, high = standardized.min(axis=0) - 0.3, standardized.max(axis=0) + 0.3
    gx, gy = np.meshgrid(np.linspace(low[0], high[0], 180), np.linspace(low[1], high[1], 180))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    scores = grid @ model_data["w"] + model_data["b"]
    ref_scores = grid @ np.array(metrics["svc"]["w"]) + metrics["svc"]["b"]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.28, top=0.96)
    ax.contour(gx, gy, scores.reshape(gx.shape), levels=[-1, 1], colors="#888888", linestyles=":", linewidths=0.8)
    ax.contour(gx, gy, scores.reshape(gx.shape), levels=[0], colors=colors["Heuristic"], linewidths=2)
    ax.contour(gx, gy, ref_scores.reshape(gx.shape), levels=[0], colors="#333333", linestyles="--", linewidths=1.3)
    for label, color in [(-1, "#4477AA"), (1, "#CC7733")]:
        mask = (labels == label) & ~is_test
        ax.scatter(*standardized[mask].T, color=color, s=13, linewidths=0, label=f"Class {label:+d}")
        mask = (labels == label) & is_test
        ax.scatter(*standardized[mask].T, color=color, s=18, marker="x", linewidths=0.8)
    support_coordinates = standardized[model_data["train_indices"]][model_data["support"]]
    ax.scatter(*support_coordinates.T, facecolors="none", edgecolors="#555555", linewidths=0.5, s=35)
    ax.set(xlabel="Standardized feature 1", ylabel="Standardized feature 2", xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False)
    export(fig, "boundary")

    # Mathematical illustration: opposite-label pair, ai-aj=0, C=1.
    # This is an illustrative feasible segment, not an empirical observation.
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.25, top=0.97)
    ax.plot([0, 1], [0, 1], color=colors["Heuristic"], lw=2.0, label="Feasible pair segment")
    ax.scatter([0.2], [0.2], color="#777777", s=35, label="Current pair", zorder=3)
    ax.scatter([0.7], [0.7], color=colors["Heuristic"], s=45, marker="s", label="Updated pair", zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xlabel(r"$\alpha_j$", fontsize=12)
    ax.set_ylabel(r"$\alpha_i$", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "C"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "C"])
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    export(fig, "pair_geometry")
    (args.output / "figure_metadata.json").write_text(json.dumps({
        "data": "synthetic", "all_samples_and_seeds_retained": True,
        "scaling": "median and min-max range of all five matched seeds; not a confidence interval",
        "ablation": "100*(1-heuristic_updates/random_updates); all five paired seeds at each size; bar=median",
        "convergence": "one seed-42 split; no smoothing; log display floor 1e-16",
        "pair_geometry": "mathematical example with opposite labels, ai=aj and C=1",
        "boundary": "all 300 observations; circles=train, crosses=test; open rings=support vectors; red=SMO, dashed gray=SVC",
        "ablation_reduction_range_percent": [min(all_reductions), max(all_reductions)],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
