"""Run ROSE vs non-ROSE workflows repeatedly; record runtimes and MSE curves (no plotting).

Writes JSON suitable for later plotting: per-trial series plus mean/std aggregates.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

_RE_ROSE_ITER = re.compile(r"^Iteration\s+(\d+):\s+([\d.eE+-]+)\s*$")
_RE_STANDARD_MSE = re.compile(r"^MSE:\s+([\d.eE+-]+)\s*$")
_RE_RUNTIME = re.compile(r"^workflow completed in ([\d.eE+-]+) seconds\s*$")


def parse_rose_mse(stdout: str) -> list[float]:
    vals: list[float] = []
    for line in stdout.splitlines():
        m = _RE_ROSE_ITER.match(line.strip())
        if m:
            vals.append(float(m.group(2)))
    return vals


def parse_standard_mse(stdout: str) -> list[float]:
    vals: list[float] = []
    for line in stdout.splitlines():
        m = _RE_STANDARD_MSE.match(line.strip())
        if m:
            vals.append(float(m.group(1)))
    return vals


def parse_reported_runtime(stdout: str) -> float | None:
    for line in stdout.splitlines():
        m = _RE_RUNTIME.match(line.strip())
        if m:
            return float(m.group(1))
    return None


def _run_driver(
    script: str,
    *,
    cwd: Path,
    python_exe: str,
) -> tuple[str, str, int, float]:
    """Return stdout, stderr, returncode, wall_seconds."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [python_exe, str(cwd / script)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    return proc.stdout, proc.stderr, proc.returncode, wall


def _summarize_runtimes(seconds: list[float]) -> dict:
    arr = np.asarray(seconds, dtype=float)
    out = {
        "n": int(arr.size),
        "mean_s": float(np.mean(arr)) if arr.size else None,
        "std_s": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_s": float(np.min(arr)) if arr.size else None,
        "max_s": float(np.max(arr)) if arr.size else None,
        "per_trial_s": [float(x) for x in seconds],
    }
    return out


def _summarize_mse_curves(series: list[list[float]]) -> dict:
    """Mean/std per iteration index over trials (common prefix length = min length)."""
    if not series:
        return {
            "n_trials": 0,
            "lengths_per_trial": [],
            "min_length": 0,
            "mean_per_iteration": [],
            "std_per_iteration": [],
            "per_trial": [],
        }
    lengths = [len(s) for s in series]
    L = min(lengths)
    if L == 0:
        return {
            "n_trials": len(series),
            "lengths_per_trial": lengths,
            "min_length": 0,
            "mean_per_iteration": [],
            "std_per_iteration": [],
            "per_trial": series,
        }
    arr = np.array([s[:L] for s in series], dtype=float)
    std_axis = np.std(arr, axis=0, ddof=1) if len(series) > 1 else np.zeros(L)
    return {
        "n_trials": len(series),
        "lengths_per_trial": lengths,
        "min_length": L,
        "mean_per_iteration": arr.mean(axis=0).tolist(),
        "std_per_iteration": std_axis.tolist(),
        "per_trial": series,
    }


def collect(
    *,
    n_runs: int,
    run_rose: bool,
    run_standard: bool,
    python_exe: str,
    cwd: Path,
) -> dict:
    """Run benchmarks; each trial runs ``clean()`` inside the child script."""
    results: dict = {
        "meta": {
            "project_root": str(cwd),
            "python": python_exe,
            "n_runs": n_runs,
        },
        "rose": {},
        "standard": {},
    }

    rose_runtimes: list[float] = []
    rose_wall: list[float] = []
    rose_mse: list[list[float]] = []

    std_runtimes: list[float] = []
    std_wall: list[float] = []
    std_mse: list[list[float]] = []

    for k in range(n_runs):
        if run_standard:
            print(f"[trial {k + 1}/{n_runs}] standard_wf.py …", flush=True)
            out, err, code, wall = _run_driver(
                "standard_wf.py", cwd=cwd, python_exe=python_exe
            )
            if code != 0:
                raise RuntimeError(
                    f"standard_wf failed (exit {code}): {err[-2000:]!r}"
                )
            std_wall.append(wall)
            std_runtimes.append(parse_reported_runtime(out) or wall)
            std_mse.append(parse_standard_mse(out))

        if run_rose:
            print(f"[trial {k + 1}/{n_runs}] workflow.py (ROSE) …", flush=True)
            out, err, code, wall = _run_driver(
                "workflow.py", cwd=cwd, python_exe=python_exe
            )
            if code != 0:
                raise RuntimeError(
                    f"workflow.py failed (exit {code}): {err[-2000:]!r}"
                )
            rose_wall.append(wall)
            rose_runtimes.append(parse_reported_runtime(out) or wall)
            rose_mse.append(parse_rose_mse(out))

    if run_standard:
        results["standard"] = {
            "wall_time_s": _summarize_runtimes(std_wall),
            "reported_timer_s": _summarize_runtimes(
                [x for x in std_runtimes if x is not None]
            ),
            "mse": _summarize_mse_curves(std_mse),
        }
    if run_rose:
        results["rose"] = {
            "wall_time_s": _summarize_runtimes(rose_wall),
            "reported_timer_s": _summarize_runtimes(
                [x for x in rose_runtimes if x is not None]
            ),
            "mse": _summarize_mse_curves(rose_mse),
        }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of full end-to-end trials per driver (default: 5)",
    )
    p.add_argument(
        "--rose-only",
        action="store_true",
        help="Only run workflow.py (ROSE)",
    )
    p.add_argument(
        "--standard-only",
        action="store_true",
        help="Only run standard_wf.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark_data.json",
        help="Where to write JSON results",
    )
    args = p.parse_args()

    run_rose = not args.standard_only
    run_standard = not args.rose_only
    if not run_rose and not run_standard:
        p.error("Select at least one of ROSE or standard (don't pass both --rose-only and --standard-only)")

    data = collect(
        n_runs=args.runs,
        run_rose=run_rose,
        run_standard=run_standard,
        python_exe=sys.executable,
        cwd=PROJECT_ROOT,
    )

    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")

    # Short human-readable summary
    for name, key in ("standard (no ROSE)", "standard"), ("ROSE", "rose"):
        block = data.get(key) or {}
        if not block:
            continue
        wt = block.get("wall_time_s", {})
        mse = block.get("mse", {})
        print(f"\n=== {name} ===")
        print(
            f"  Wall time (perf_counter): mean={wt.get('mean_s')} s, "
            f"std={wt.get('std_s')} s, n={wt.get('n')}"
        )
        if mse.get("mean_per_iteration"):
            print(
                f"  MSE curve: {mse.get('min_length')} iterations summarized, "
                f"final mean MSE={mse['mean_per_iteration'][-1]:.6g} "
                f"(std over trials={mse['std_per_iteration'][-1]:.6g})"
            )


if __name__ == "__main__":
    main()
