from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("benchmark_data.json")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_rmse_panel(
    ax,
    block: dict | None,
    *,
    title: str,
    color: str,
    marker: str,
) -> None:
    """Mean RMSE ± std over trials vs iteration (same style as previous combined plot)."""
    if not block:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    mse = block.get("mse", {})
    mean = mse.get("mean_per_iteration")
    std = mse.get("std_per_iteration")
    if not mean:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    iters = np.arange(len(mean))
    mean = np.asarray(mean)
    std = np.asarray(std)
    rmse = np.sqrt(mean)
    lo = np.sqrt(np.maximum(mean - std, 0.0))
    hi = np.sqrt(mean + std)

    ax.plot(iters, rmse, label=title, color=color, marker=marker)
    ax.fill_between(iters, lo, hi, alpha=0.2, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def main() -> None:
    data = load_data(DATA_PATH)

    fig, (ax_rose, ax_std) = plt.subplots(1, 2, figsize=(10, 5))

    plot_rmse_panel(ax_rose, data.get("rose"), title="ROSE", color="tab:blue", marker="o")
    plot_rmse_panel(
        ax_std,
        data.get("standard"),
        title="standard (no ROSE)",
        color="tab:orange",
        marker="s",
    )
    ax_std.set_ylabel("")
    ax_std.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.suptitle("ROSE vs standard (10 runs, mean ± std)", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "benchmark_side_by_side.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
