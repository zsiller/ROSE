#!/usr/bin/env python
"""Batch MCMC method-comparison study for the Sod shock-tube inverse problem.

Drives many Dakota ``bayes_calibration`` chains over the SAME inverse problem
(infer the 4 Sod ICs from final-time density observations) so their samplers,
proposal-covariance strategies and gradient usage can be compared apples to
apples. Each chain runs in its own isolated working directory under
``dakota_mcmc/experiments/<name>/`` so that many can run concurrently without
clobbering each other's Dakota scratch files (``params.in``, ``results.out``,
``QuesoDiagnostics/``, restart, tabular, exported chain).

For every run it records:
  * the rendered Dakota deck (``<name>.in``) and Dakota stdout/stderr,
  * the exported chain (``chain.dat``) and a tidy ``chain_params.npy`` (N x 4),
  * acceptance rate (computed from the chain, cross-checked against QUESO's
    reported rejection percentage when available),
  * posterior marginals/traces (``<name>_marginals.png``) and the reconstructed
    density field band (``<name>_field.png``),
  * ``metadata.json`` (config + posterior summary + timings).

The matrix has three groups (see ``build_matrix``):
  1. QUESO core samplers x proposal strategy  -> standardized-space vs.
     prior-covariance vs. an ensemble "initial covariance matrix".
  2. gradient-informed samplers (MUQ MALA / DILI, QUESO derivative proposal).

Examples
--------
    source ../rose_env/bin/activate
    # validate every deck quickly (tiny chains, exact forward):
    python dakota_mcmc/run_experiments.py --smoke
    # the real overnight study (euler forward, parallel):
    python dakota_mcmc/run_experiments.py --jobs 8
    # just write the decks, run nothing:
    python dakota_mcmc/run_experiments.py --dry-run
    # a subset:
    python dakota_mcmc/run_experiments.py --only dram__prior_std mala_muq
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    N_FIELD, PARAM_NAMES, T_FINAL, TRUTH, exact_density_on_cells,
    make_forward, make_plots,
)
from run_mcmc import (  # noqa: E402
    LOWER_BOUNDS, UPPER_BOUNDS, MethodSpec, calibration_terms_from_obs,
    load_chain, load_obs_densities, resolve_dakota, write_dakota_input_ex,
)

EXP_DIR = HERE / "experiments"
DEFAULT_OBS = HERE / "sod_obs.dat"
DEFAULT_CELLS = HERE / "sod_cells.npy"
DEFAULT_PROP_COV = HERE / "prop_cov.dat"
DRIVER = HERE / "sod_driver.sh"

# Honest starting point for every chain: the prior centre, NOT the truth the
# observations were generated at (TRUTH lives in common). Keeping the start away
# from the answer makes convergence/acceptance comparisons meaningful.
PRIOR_CENTER = np.array([1.0e5, 1.0e4, 1.0, 0.125])
# Truth behind the staged observations (see README / write_observations example).
OBS_TRUTH = np.array([0.9e5, 0.9e4, 0.9, 0.1])


@dataclass
class RunConfig:
    """One row of the study: a MethodSpec plus run-level bookkeeping."""
    name: str
    group: str
    spec: MethodSpec
    chain_samples: int
    seed: int
    forward: str
    note: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["spec"] = asdict(self.spec)
        return d


# --------------------------------------------------------------------------- #
# The run matrix
# --------------------------------------------------------------------------- #
# Non-adaptive samplers mix slowly, so they get longer chains than the adaptive
# ones; gradient runs are dominated by finite-difference cost, so they get the
# shortest. All scaled by --scale and overridable per group.
BASE_SAMPLES = {
    "metropolis_hastings": 15000,
    "delayed_rejection": 15000,
    "adaptive_metropolis": 12000,
    "dram": 12000,
    "gradient": 6000,
    "scale": 10000,
}

# Covariance multipliers for the prior-proposal scaling sweep. The 'proposal
# prior' covariance is ~10-66x too wide (see README/analysis); these shrink it.
# In standardized space the prior is a unit box (proposal cov = diag(0.333),
# std ~0.577 per dim). A validation sweep [0.2,0.05,0.02,0.005] gave acceptance
# [0.010,0.017,0.050,0.175] -- monotonic but still below the ~0.234 ideal even at
# 0.005, because rho_high is pinned ~66x tighter than the prior and a single
# scalar can't match that anisotropy. So the scalar optimum sits BELOW 0.005;
# the default brackets it down to 0.001 (the slow-mixing, high-acceptance end).
DEFAULT_MULTIPLIERS = [0.1, 0.02, 0.005, 0.001]

# (display name, QUESO sampler keyword)
_CORE_SAMPLERS = [
    ("mh", "metropolis_hastings"),
    ("am", "adaptive_metropolis"),
    ("dr", "delayed_rejection"),
    ("dram", "dram"),
]


def build_matrix(*, scale: float, forward: str, seed0: int,
                 prop_cov: Path, include_muq: bool = False,
                 multipliers: list[float] | None = None) -> list[RunConfig]:
    runs: list[RunConfig] = []
    seed = seed0

    def n(key: str) -> int:
        return max(50, int(round(BASE_SAMPLES[key] * scale)))

    # --- Group 1: QUESO core samplers x proposal strategy --------------------
    # Three "how do we build/scale the proposal" strategies:
    #   prior_native : prior covariance, sampled in native (physical) units
    #   prior_std    : prior covariance, sampled in standardized space
    #   ens_file     : an empirical "initial covariance matrix" read from the
    #                  EnKF ensemble (prop_cov.dat), native units
    strategies = [
        ("prior_native", dict(standardized_space=False, proposal="prior"),
         "prior cov, native units"),
        ("prior_std", dict(standardized_space=True, proposal="prior"),
         "prior cov, standardized space"),
        ("ens_file", dict(standardized_space=False, proposal="file",
                          proposal_file=str(prop_cov)),
         "EnKF ensemble covariance (initial cov matrix), native units"),
    ]
    for sname, sampler in _CORE_SAMPLERS:
        for tag, kw, note in strategies:
            spec = MethodSpec(library="queso", sampler=sampler,
                              gradients="none", **kw)
            runs.append(RunConfig(
                name=f"{sname}__{tag}", group="core", spec=spec,
                chain_samples=n(sampler), seed=seed, forward=forward,
                note=f"{sampler} | {note}"))
            seed += 1

    # --- Group 1b: prior-proposal scaling sweep ------------------------------
    # Shrink the (too-wide) prior proposal by a scalar covariance multiplier in
    # standardized space, on non-adaptive metropolis_hastings so acceptance
    # reflects the proposal scale directly (no adaptation confound). Brackets the
    # predicted near-optimal multiplier ~0.02.
    for mult in (multipliers if multipliers is not None else DEFAULT_MULTIPLIERS):
        spec = MethodSpec(library="queso", sampler="metropolis_hastings",
                          standardized_space=True, proposal="prior",
                          proposal_multiplier=mult, gradients="none")
        runs.append(RunConfig(
            name=f"scale_mh_m{mult:g}", group="scale", spec=spec,
            chain_samples=n("scale"), seed=seed, forward=forward,
            note=f"metropolis_hastings | prior cov x {mult:g}, standardized "
                 f"space (proposal-scale sweep)"))
        seed += 1

    # --- Group 2: gradient-informed samplers ---------------------------------
    # This Dakota build is compiled WITHOUT MUQ, so MALA/DILI/HMC are
    # unavailable; the gradient route that DOES run is QUESO's
    # ``proposal_covariance derivatives`` -- it finite-differences the 16
    # calibration residuals into a Jacobian, forms the Gauss-Newton Hessian of
    # the misfit, and uses its inverse as a local Gaussian (Laplace/Newton)
    # proposal, refreshed every ``update_period`` steps. That is the
    # gradient/Hessian-driven proposal analogue available here.
    grad_runs = [
        ("deriv_mh_std",
         MethodSpec(library="queso", sampler="metropolis_hastings",
                    standardized_space=True, proposal="derivatives",
                    proposal_update_period=200, gradients="numerical",
                    fd_interval="central", fd_step=1e-3),
         "QUESO MH, gradient/Hessian-built (derivative) proposal, standardized "
         "space, FD gradients"),
        ("deriv_dram_std",
         MethodSpec(library="queso", sampler="dram",
                    standardized_space=True, proposal="derivatives",
                    proposal_update_period=200, gradients="numerical",
                    fd_interval="central", fd_step=1e-3),
         "QUESO DRAM on top of a gradient/Hessian-built derivative proposal, "
         "standardized space, FD gradients"),
        ("deriv_mh_native",
         MethodSpec(library="queso", sampler="metropolis_hastings",
                    standardized_space=False, proposal="derivatives",
                    proposal_update_period=200, gradients="numerical",
                    fd_interval="central", fd_step=1e-3),
         "QUESO MH, gradient/Hessian-built derivative proposal, native space "
         "(does gradient info remove the need for standardization?)"),
    ]
    for gname, spec, note in grad_runs:
        runs.append(RunConfig(name=gname, group="gradient", spec=spec,
                              chain_samples=n("gradient"), seed=seed,
                              forward=forward, note=note))
        seed += 1

    # --- Optional MUQ samplers (need a MUQ-enabled Dakota rebuild) ------------
    # Kept here so they "just work" once Dakota is built with MUQ; opt in with
    # --include-muq. MUQ lacks standardized_space, hence the prior-cov metric.
    if include_muq:
        runs.append(RunConfig(
            name="mala_muq", group="gradient_muq",
            spec=MethodSpec(library="muq", sampler="mala",
                            standardized_space=False, proposal="prior",
                            gradients="numerical", fd_interval="central",
                            fd_step=1e-3, sampler_opts={"step_size": 0.3}),
            chain_samples=n("gradient"), seed=seed, forward=forward,
            note="MUQ Metropolis-adjusted Langevin (true gradient/HMC-like); "
                 "REQUIRES a MUQ-enabled Dakota build"))
        seed += 1
        runs.append(RunConfig(
            name="dili_muq", group="gradient_muq",
            spec=MethodSpec(library="muq", sampler="dili",
                            standardized_space=False, proposal="prior",
                            gradients="numerical", fd_interval="central",
                            fd_step=1e-3),
            chain_samples=n("gradient"), seed=seed, forward=forward,
            note="MUQ dimension-independent likelihood-informed (Hessian); "
                 "REQUIRES a MUQ-enabled Dakota build"))
        seed += 1

    return runs


# --------------------------------------------------------------------------- #
# Per-run execution (worker side: NO matplotlib -- plotting is done serially in
# the main thread afterwards, since pyplot is not thread-safe).
# --------------------------------------------------------------------------- #
def acceptance_from_chain(params: np.ndarray) -> float:
    """Fraction of steps the chain actually moved -- a sampler-agnostic accept
    rate (counts both first- and delayed-rejection acceptances for DR/DRAM)."""
    if params.shape[0] < 2:
        return float("nan")
    moved = np.any(np.diff(params, axis=0) != 0.0, axis=1)
    return float(moved.mean())


def queso_reported_acceptance(run_dir: Path) -> float | None:
    """Last 'current rejection percentage = X %' QUESO printed, as accept rate."""
    f = run_dir / "QuesoDiagnostics" / "display_sub0.txt"
    if not f.is_file():
        return None
    last = None
    for line in f.read_text().splitlines():
        if "rejection percentage" in line:
            try:
                last = float(line.split("=")[-1].strip().rstrip("%").strip())
            except ValueError:
                pass
    return None if last is None else 1.0 - last / 100.0


def load_param_chain(path: Path) -> np.ndarray:
    """Read the 4 Sod-param columns from any Dakota export (QUESO or MUQ).

    Falls back to a positional read if the header does not name the params."""
    try:
        chain = load_chain(path)
        if chain.shape[1] == 4:
            return chain
    except (ValueError, KeyError, IndexError):
        pass
    with open(path) as fh:
        first = fh.readline()
    skip = 1 if first.lstrip().startswith("%") else 0
    data = np.atleast_2d(np.loadtxt(path, skiprows=skip))
    # Columns are [id, interface, p_high, p_low, rho_high, rho_low, <responses>].
    if data.shape[1] >= 6:
        return data[:, 2:6]
    raise ValueError(f"cannot locate 4 param columns in {path} "
                     f"(shape {data.shape})")


def run_one(cfg: RunConfig, *, dakota: Path, obs: Path, cells: Path,
            dry: bool) -> dict:
    """Render the deck, run Dakota in an isolated dir, save chain + metadata."""
    run_dir = EXP_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Every artifact carries the trial name so that, even though each chain
    # already runs in its own directory, no two parallel runs can ever share a
    # chain / input / output / tabular filename.
    input_path = run_dir / f"{cfg.name}.in"
    out_name = f"{cfg.name}.out"
    chain_name = f"{cfg.name}_chain.dat"
    tabular_name = f"{cfg.name}_tabular.dat"
    log_path = run_dir / f"{cfg.name}.log"
    chain_npy = run_dir / f"{cfg.name}_chain.npy"
    m = calibration_terms_from_obs(obs)

    write_dakota_input_ex(
        spec=cfg.spec, initial_point=PRIOR_CENTER, n_calibration=m,
        chain_samples=cfg.chain_samples, seed=cfg.seed, input_path=input_path,
        chain_file=chain_name, tabular_file=tabular_name,
        obs_file=str(obs.resolve()), driver=str(DRIVER.resolve()),
        lower=LOWER_BOUNDS, upper=UPPER_BOUNDS)

    rec = {"name": cfg.name, "group": cfg.group, "config": cfg.to_json(),
           "n_samples": cfg.chain_samples, "forward": cfg.forward,
           "run_dir": str(run_dir), "input_file": str(input_path),
           "output_file": str(run_dir / out_name),
           "chain_file": str(run_dir / chain_name),
           "chain_npy": str(chain_npy)}
    if dry:
        rec["status"] = "written"
        return rec

    env = os.environ.copy()
    env["SOD_FORWARD"] = cfg.forward
    env["SOD_CELLS"] = str(cells.resolve())
    cmd = [str(dakota), "-i", input_path.name, "-o", out_name]
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=run_dir, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    rec["runtime_s"] = round(runtime, 1)
    rec["returncode"] = proc.returncode

    chain_path = run_dir / chain_name
    if proc.returncode != 0 or not chain_path.is_file():
        rec["status"] = "FAILED"
        rec["error"] = (f"dakota rc={proc.returncode}; "
                        f"chain {'missing' if not chain_path.is_file() else 'present'}"
                        f" -- see {log_path}")
        return rec

    params = load_param_chain(chain_path)
    np.save(chain_npy, params)
    rec["ms_per_sample"] = round(1000.0 * runtime / max(params.shape[0], 1), 3)

    burn = int(cfg.spec.burn_in_frac * params.shape[0])
    post = params[burn:] if params.shape[0] > burn else params
    rec.update({
        "status": "ok",
        "n_chain_rows": int(params.shape[0]),
        "burn": int(burn),
        "acc_rate": acceptance_from_chain(params),
        "acc_rate_post": acceptance_from_chain(post),
        "acc_queso_reported": queso_reported_acceptance(run_dir),
        "post_mean": post.mean(axis=0).tolist(),
        "post_std": post.std(axis=0).tolist(),
        "post_q025": np.percentile(post, 2.5, axis=0).tolist(),
        "post_q975": np.percentile(post, 97.5, axis=0).tolist(),
    })
    return rec


# --------------------------------------------------------------------------- #
# Post-processing (main thread): field reconstruction + plots + metadata.json
# --------------------------------------------------------------------------- #
def postprocess(rec: dict, *, cells: Path, obs: Path, truth: np.ndarray,
                field_draws: int, seed: int) -> dict:
    run_dir = Path(rec["run_dir"])
    meta_path = run_dir / f"{rec['name']}_metadata.json"
    if rec.get("status") != "ok":
        meta_path.write_text(json.dumps(rec, indent=2))
        return rec

    params = np.load(rec["chain_npy"])
    burn = rec["burn"]
    post = params[burn:] if params.shape[0] > burn else params

    cell_idx = np.load(cells)
    obs_y = load_obs_densities(obs)
    mean = post.mean(axis=0)
    quantiles = np.percentile(post, [2.5, 50, 97.5], axis=0)

    g_full = make_forward(rec["forward"], in_process=True)
    rng = np.random.default_rng(seed)
    n_draw = min(field_draws, post.shape[0])
    idx = rng.choice(post.shape[0], size=n_draw, replace=False)
    fields = np.array([g_full(post[j]) for j in idx])
    field_mean = fields.mean(axis=0)
    field_lo, field_hi = np.percentile(fields, [2.5, 97.5], axis=0)
    exact_rho = exact_density_on_cells(truth, T_FINAL, N_FIELD)
    post_rmse = float(np.sqrt(np.mean((field_mean - exact_rho) ** 2)))
    rec["post_rmse"] = post_rmse

    res = {
        "chain": params, "post": post, "truth": truth, "mean": mean,
        "std": post.std(axis=0), "quantiles": quantiles,
        "acc_rate": rec["acc_rate"],
        "x_cells": np.arange(N_FIELD) / N_FIELD, "exact": exact_rho,
        "field_mean": field_mean, "field_lo": field_lo, "field_hi": field_hi,
        "obs_x": cell_idx / N_FIELD, "obs_y": obs_y, "post_rmse": post_rmse,
        "burn": burn, "forward": rec["forward"],
    }
    make_plots(res, run_dir, label=rec["name"])
    meta_path.write_text(json.dumps(rec, indent=2))
    write_run_txt(rec, run_dir / f"{rec['name']}_result.txt")
    return rec


def _proposal_str(spec: dict) -> str:
    kind = spec.get("proposal")
    if kind == "file":
        return f"file ({spec.get('proposal_file')}) matrix"
    if kind == "derivatives":
        return f"derivatives (update_period={spec.get('proposal_update_period')})"
    if kind == "prior":
        mult = spec.get("proposal_multiplier")
        return "prior" if mult is None else f"prior (multiplier={mult})"
    return str(kind)


def _trial_block(rec: dict) -> str:
    """Human-readable block: trial name, full configuration, acceptance, time."""
    spec = rec.get("config", {}).get("spec", {})
    lines = [
        "=" * 72,
        f"TRIAL: {rec['name']}   [group: {rec.get('group', '?')}]",
        "-" * 72,
        "  CONFIGURATION",
        f"    library             : {spec.get('library')}",
        f"    sampler             : {spec.get('sampler')}",
        f"    standardized_space  : {spec.get('standardized_space')}"
        f"{'  (QUESO only; ignored for muq)' if spec.get('library') == 'muq' else ''}",
        f"    proposal_covariance : {_proposal_str(spec)}",
        f"    gradients           : {spec.get('gradients')}"
        + (f" ({spec.get('fd_interval')} FD, step={spec.get('fd_step')})"
           if spec.get('gradients') == 'numerical' else ""),
    ]
    if spec.get("sampler_opts"):
        lines.append(f"    sampler_opts        : {spec.get('sampler_opts')}")
    lines += [
        f"    chain_samples       : {rec.get('n_samples')}",
        f"    burn_in             : {rec.get('burn', int(spec.get('burn_in_frac', 0.2) * (rec.get('n_samples') or 0)))}",
        f"    seed                : {rec.get('config', {}).get('seed')}",
        f"    forward model       : {rec.get('forward')}",
        f"    description         : {rec.get('config', {}).get('note', '')}",
        "  RESULTS",
        f"    status              : {rec.get('status', '?')}",
    ]
    if rec.get("status") == "ok":
        mean = rec.get("post_mean", [])
        std = rec.get("post_std", [])
        lines += [
            f"    acceptance (chain)  : {_fmt(rec.get('acc_rate'), '.4f')}",
            f"    acceptance (post-burn): {_fmt(rec.get('acc_rate_post'), '.4f')}"
            "   <- representative for adaptive samplers (am/dram)",
            f"    acceptance (QUESO)  : {_fmt(rec.get('acc_queso_reported'), '.4f')}",
            f"    runtime             : {_fmt(rec.get('runtime_s'), '.1f')} s"
            f"   ({_fmt(rec.get('ms_per_sample'), '.2f')} ms/sample)",
            f"    posterior RMSE      : {_fmt(rec.get('post_rmse'), '.4e')}",
            "    posterior mean (+/- std) vs truth:",
        ]
        for k, nm in enumerate(PARAM_NAMES):
            if k < len(mean):
                lines.append(
                    f"        {nm:9s} {mean[k]:.5e} +/- "
                    f"{(std[k] if k < len(std) else float('nan')):.2e}  "
                    f"(truth {OBS_TRUTH[k]:.4e})")
        lines += [
            f"    chain (Dakota)      : {rec.get('chain_file')}",
            f"    chain (numpy Nx4)   : {rec.get('chain_npy')}",
        ]
    else:
        lines += [
            f"    runtime             : {_fmt(rec.get('runtime_s'), '.1f')} s",
            f"    error               : {rec.get('error', rec.get('postprocess_error', 'unknown'))}",
        ]
    return "\n".join(lines)


def write_run_txt(rec: dict, path: Path) -> None:
    path.write_text(_trial_block(rec) + "\n")


def write_results_txt(records: list[dict], out_dir: Path) -> None:
    """One combined, human-readable results.txt: config + acceptance + time."""
    header = [
        "Sod shock-tube MCMC method comparison -- results",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"forward model: {records[0].get('forward') if records else '?'}",
        f"observations truth: {OBS_TRUTH.tolist()}  "
        f"chain start (prior centre): {PRIOR_CENTER.tolist()}",
        "",
        "Quick table (acceptance = fraction of accepted MCMC steps; "
        "time = Dakota wall-clock):",
        "",
        f"  {'trial':18s} {'sampler':20s} {'accept':>8s} {'acc_post':>8s} "
        f"{'time[s]':>9s} {'ms/samp':>8s} {'RMSE':>10s}",
        f"  {'-'*18:18s} {'-'*20:20s} {'-'*8:>8s} {'-'*8:>8s} {'-'*9:>9s} "
        f"{'-'*8:>8s} {'-'*10:>10s}",
    ]
    for r in records:
        spec = r.get("config", {}).get("spec", {})
        header.append(
            f"  {r['name']:18s} {str(spec.get('sampler')):20s} "
            f"{_fmt(r.get('acc_rate'), '.3f'):>8s} "
            f"{_fmt(r.get('acc_rate_post'), '.3f'):>8s} "
            f"{_fmt(r.get('runtime_s'), '.0f'):>9s} "
            f"{_fmt(r.get('ms_per_sample'), '.2f'):>8s} "
            f"{_fmt(r.get('post_rmse'), '.2e'):>10s}")
    blocks = [_trial_block(r) for r in records]
    (out_dir / "results.txt").write_text(
        "\n".join(header) + "\n\n" + "\n".join(blocks) + "\n")


def write_summary(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    cols = ["name", "group", "status", "n_samples", "acc_rate", "acc_rate_post",
            "acc_queso_reported", "post_rmse", "runtime_s", "ms_per_sample"]
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join(_csv(r.get(c)) for c in cols))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(records, indent=2))

    # Markdown table
    md = ["# MCMC method-comparison summary", "",
          "| run | group | status | N | acc (chain) | acc (QUESO) | "
          "post RMSE | runtime |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        md.append("| {name} | {group} | {status} | {N} | {acc} | {accq} | "
                  "{rmse} | {rt} |".format(
                      name=r["name"], group=r.get("group", ""),
                      status=r.get("status", "?"), N=r.get("n_samples", ""),
                      acc=_fmt(r.get("acc_rate"), ".3f"),
                      accq=_fmt(r.get("acc_queso_reported"), ".3f"),
                      rmse=_fmt(r.get("post_rmse"), ".2e"),
                      rt=_fmt(r.get("runtime_s"), ".0f")))
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")


def _csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fmt(v, spec) -> str:
    return "-" if v is None else format(v, spec)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel chains (default: ncpu-2)")
    ap.add_argument("--forward", choices=("exact", "euler", "surrogate"),
                    default="euler", help="forward model for every chain")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every chain length by this factor")
    ap.add_argument("--seed0", type=int, default=12345,
                    help="seed for the first run; each subsequent run +1")
    ap.add_argument("--obs", type=Path, default=DEFAULT_OBS)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--prop-cov", dest="prop_cov", type=Path, default=DEFAULT_PROP_COV,
                    help="ensemble 'initial covariance matrix' for the ens_file runs")
    ap.add_argument("--truth", type=float, nargs=4, default=list(OBS_TRUTH),
                    metavar=tuple(PARAM_NAMES), help="truth for plot overlays")
    ap.add_argument("--include-muq", dest="include_muq", action="store_true",
                    help="also queue MUQ MALA/DILI runs (needs a MUQ-enabled "
                         "Dakota build; this build is QUESO-only)")
    ap.add_argument("--prior-multipliers", dest="multipliers", type=float,
                    nargs="+", default=None,
                    help="covariance multipliers for the prior-proposal scaling "
                         f"sweep (default: {DEFAULT_MULTIPLIERS})")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these run names (default: all)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny chains + exact forward to validate every deck")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="write decks only; do not run Dakota")
    ap.add_argument("--field-draws", dest="field_draws", type=int, default=120,
                    help="posterior draws for the reconstructed-field band")
    ap.add_argument("--tag", default=None,
                    help="write outputs to experiments_<tag>/ instead of "
                         "experiments/ (isolate one-off cases, e.g. --tag noiseless)")
    ap.add_argument("--dakota", type=Path, default=None)
    args = ap.parse_args()

    if args.tag:
        global EXP_DIR
        EXP_DIR = HERE / f"experiments_{args.tag}"

    forward = "exact" if args.smoke else args.forward
    scale = (40.0 / BASE_SAMPLES["gradient"]) if args.smoke else args.scale
    truth = np.asarray(args.truth, dtype=float)

    if not args.obs.is_file():
        sys.exit(f"obs file not found: {args.obs} (run write_observations.py first)")
    if not args.cells.is_file():
        sys.exit(f"cells file not found: {args.cells}")
    if not args.prop_cov.is_file():
        sys.exit(f"prop_cov not found: {args.prop_cov} (run gen_proposal_cov.py)")

    runs = build_matrix(scale=scale, forward=forward, seed0=args.seed0,
                        prop_cov=args.prop_cov.resolve(),
                        include_muq=args.include_muq, multipliers=args.multipliers)
    if args.only:
        keep = set(args.only)
        runs = [r for r in runs if r.name in keep]
        missing = keep - {r.name for r in runs}
        if missing:
            avail = [r.name for r in build_matrix(
                scale=1, forward=forward, seed0=0, prop_cov=args.prop_cov,
                include_muq=True, multipliers=args.multipliers)]
            sys.exit(f"unknown run name(s): {sorted(missing)}\n"
                     f"available: {avail}")
    if not runs:
        sys.exit("no runs selected")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    dakota = None if args.dry_run else resolve_dakota(args.dakota)

    print(f"[study] {len(runs)} runs | forward={forward} | jobs={args.jobs} | "
          f"smoke={args.smoke} | dry_run={args.dry_run}")
    for r in runs:
        print(f"   - {r.name:18s} {r.spec.library}/{r.spec.sampler:20s} "
              f"N={r.chain_samples:6d}  {r.note}")

    records: list[dict] = []
    if args.dry_run:
        for r in runs:
            records.append(run_one(r, dakota=Path("dakota"), obs=args.obs,
                                   cells=args.cells, dry=True))
        print(f"[done] wrote {len(records)} decks under {EXP_DIR} (no runs)")
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, r, dakota=dakota, obs=args.obs,
                            cells=args.cells, dry=False): r for r in runs}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"name": r.name, "group": r.group, "status": "FAILED",
                       "error": repr(exc), "run_dir": str(EXP_DIR / r.name)}
            status = rec.get("status", "?")
            acc = _fmt(rec.get("acc_rate"), ".3f")
            print(f"[run ] {r.name:18s} {status:8s} acc={acc} "
                  f"({rec.get('runtime_s', '?')}s)")
            records.append(rec)

    print(f"[post] reconstructing fields + plotting ({len(records)} runs)...")
    by_name = {r.name: r for r in runs}
    done = []
    for rec in records:
        try:
            done.append(postprocess(rec, cells=args.cells, obs=args.obs,
                                    truth=truth, field_draws=args.field_draws,
                                    seed=by_name[rec["name"]].seed))
        except Exception as exc:  # noqa: BLE001
            rec["postprocess_error"] = repr(exc)
            done.append(rec)
            print(f"[post] {rec['name']}: plotting failed: {exc!r}")

    done.sort(key=lambda r: (r.get("group", ""), r["name"]))
    write_summary(done, EXP_DIR)
    write_results_txt(done, EXP_DIR)
    ok = sum(1 for r in done if r.get("status") == "ok")
    print(f"[done] {ok}/{len(done)} ok in {time.time() - t0:.0f}s | "
          f"results: {EXP_DIR/'results.txt'} | summary: {EXP_DIR/'summary.md'}")


if __name__ == "__main__":
    main()
