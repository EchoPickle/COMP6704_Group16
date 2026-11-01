from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCALES = ["S", "M", "L"]
KERNELS = ["Linear", "RBF", "Polynomial"]
ACCURACY = {
    "Linear": [0.5793, 0.5896, 0.5829],
    "RBF": [0.5469, 0.5473, 0.5485],
    "Polynomial": [0.5687, 0.5763, 0.5756],
}
RUNTIME_SECONDS = {
    "Linear": [446.3070, 1794.5769, 4517.3597],
    "RBF": [451.6788, 1822.6228, 4584.1090],
    "Polynomial": [393.8699, 1777.5047, 4562.0442],
}
BASELINE = 0.5468


def main():
    output_dir = Path("outputs/ipm")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ipm_experiment_summary.png"

    x = np.arange(len(SCALES))
    width = 0.22
    colors = {
        "Linear": "#4C78A8",
        "RBF": "#F58518",
        "Polynomial": "#54A24B",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for idx, kernel in enumerate(KERNELS):
        offset = (idx - 1) * width
        ax1.bar(x + offset, ACCURACY[kernel], width=width, label=kernel, color=colors[kernel])
        ax2.bar(
            x + offset,
            np.array(RUNTIME_SECONDS[kernel]) / 60.0,
            width=width,
            label=kernel,
            color=colors[kernel],
        )

    ax1.axhline(BASELINE, color="crimson", linestyle="--", linewidth=1.5, label="Baseline")
    ax1.set_title("IPM Accuracy by Scale and Kernel")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SCALES)
    ax1.set_xlabel("Training subset")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.53, 0.61)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.set_title("IPM Runtime by Scale and Kernel")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SCALES)
    ax2.set_xlabel("Training subset")
    ax2.set_ylabel("Runtime (minutes)")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(output_path)


if __name__ == "__main__":
    main()
