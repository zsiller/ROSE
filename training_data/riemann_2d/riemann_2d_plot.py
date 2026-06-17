#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _preview_array(arr: np.ndarray, max_items: int = 10) -> str:
    flat = arr.reshape(-1)
    sample = flat[:max_items]
    return np.array2string(sample, precision=5, separator=", ")


def inspect_h5(path_str: str) -> None:
    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"Error: file does not exist -> {path}")
        return
    if not path.is_file():
        print(f"Error: not a file -> {path}")
        return

    stat = path.stat()
    print("=== HDF5 File Details ===")
    print(f"Path: {path.resolve()}")
    print(f"Size: {stat.st_size} bytes")
    print(f"Last Modified: {_fmt_ts(stat.st_mtime)}")
    print(f"Last Accessed: {_fmt_ts(stat.st_atime)}")
    print()

    with h5py.File(path, "r") as f:
        print("=== Root Attributes ===")
        if len(f.attrs) == 0:
            print("(none)")
        else:
            for key, value in f.attrs.items():
                print(f"- {key}: {value}")
        print()

        print("=== Datasets ===")
        datasets = []
        f.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)

        if not datasets:
            print("(no datasets)")
            return

        for name, ds in datasets:
            data = np.asarray(ds)
            print(f"- {name}")
            print(f"  shape: {data.shape}")
            print(f"  dtype: {data.dtype}")
            print(f"  preview: {_preview_array(data)}")


def _extract_2d_grid(data: np.ndarray, timestep_1_based: int) -> np.ndarray:
    timestep_idx = timestep_1_based - 1

    if data.ndim == 2 and data.shape == (128, 128):
        return data

    if data.ndim == 3:
        # Common layout: [time, x, y]
        if data.shape[1:] == (128, 128):
            if timestep_idx >= data.shape[0]:
                raise ValueError(
                    f"Requested timestep {timestep_1_based} but dataset has {data.shape[0]} time steps"
                )
            return data[timestep_idx]

        # Alternate layout: [x, y, time]
        if data.shape[:2] == (128, 128):
            if timestep_idx >= data.shape[2]:
                raise ValueError(
                    f"Requested timestep {timestep_1_based} but dataset has {data.shape[2]} time steps"
                )
            return data[:, :, timestep_idx]

    if data.ndim == 2 and 128 * 128 in data.shape:
        # Handle flattened layouts: [time, 16384] or [16384, time]
        if data.shape[1] == 128 * 128:
            if timestep_idx >= data.shape[0]:
                raise ValueError(
                    f"Requested timestep {timestep_1_based} but dataset has {data.shape[0]} time steps"
                )
            return data[timestep_idx].reshape(128, 128)
        if data.shape[0] == 128 * 128:
            if timestep_idx >= data.shape[1]:
                raise ValueError(
                    f"Requested timestep {timestep_1_based} but dataset has {data.shape[1]} time steps"
                )
            return data[:, timestep_idx].reshape(128, 128)

    raise ValueError(
        f"Unsupported dataset shape {data.shape}; expected 128x128 with optional time axis"
    )


def plot_128x128_grid(path_str: str, dataset_name: str, timestep: int) -> None:
    if timestep < 1 or timestep > 12:
        raise ValueError(f"timestep must be between 1 and 12, got {timestep}")

    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            available = [name for name, obj in f.items() if isinstance(obj, h5py.Dataset)]
            raise KeyError(
                f"Dataset '{dataset_name}' not found. Available top-level datasets: {available}"
            )

        data = np.asarray(f[dataset_name])
        grid = _extract_2d_grid(data, timestep)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(im, label=dataset_name)
    plt.title(f"{dataset_name} at timestep {timestep}")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.tight_layout()
    plt.savefig(f"riemann_plot_{timestep}_{dataset_name}_{path.stem}.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 file and/or plot a 128x128 dataset grid."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="training_data/riemann_2d/H5_files/T1.h5",
        help="Path to HDF5 file (default: training_data/riemann_2d/H5_files/T1.h5)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a 128x128 grid for the selected dataset and timestep.",
    )
    parser.add_argument(
        "--dataset",
        default="u",
        help="Dataset name to plot (example: u, rho, pressure).",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=1,
        help="Time step to plot (1 through 12).",
    )
    args = parser.parse_args()
    inspect_h5(args.path)
    if args.plot:
        plot_128x128_grid(args.path, args.dataset, args.timestep)


if __name__ == "__main__":
    main()

