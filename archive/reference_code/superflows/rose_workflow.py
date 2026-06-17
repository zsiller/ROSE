import asyncio
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner

from simulations.CDR_1D.sim_wrapper import run_cdr
from surrogate.POD import POD
from surrogate.train import Surrogate

logger = logging.getLogger(__name__)

N_INITIAL = 2
N_PER_ITER = 2
N_COMPONENTS = 5
N_TEST = 50
BETA_RANGE = (0.0, 10.0)
T_FINAL = 0.25
CANDIDATE_SIZE = 200
MAX_ITERATIONS = 8


def _run_cdr_batch(betas):
    """Run CDR for a list of betas, collect all time snapshots."""
    X_pod_rows, X_gp_rows = [], []
    x_grid = None
    for beta in betas:
        df, _ = run_cdr(theta=float(beta), t_final=T_FINAL, write_every=1000)
        if x_grid is None:
            x_grid = df.index.to_numpy()
        for t in df.columns.to_numpy():
            X_pod_rows.append(df[t].to_numpy().reshape(1, -1))
            X_gp_rows.append([float(beta), float(t)])
    return X_pod_rows, X_gp_rows, x_grid


def _make_edges(centers, lo=None, hi=None):
    edges = np.concatenate([
        [centers[0] - 0.5 * (centers[1] - centers[0])],
        0.5 * (centers[:-1] + centers[1:]),
        [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
    ])
    if lo is not None:
        edges[0] = max(edges[0], lo)
    if hi is not None:
        edges[-1] = min(edges[-1], hi)
    return edges


async def main():
    rng = np.random.default_rng(42)

    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)
    init_default_logger(logging.INFO)

    # ================================================================
    # SIMULATION TASK
    # ================================================================
    @acl.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs) -> dict:
        prev = args[0] if args else {}
        new_betas = prev.get("new_betas", None)

        if new_betas is None:
            new_betas = np.sort(rng.uniform(*BETA_RANGE, size=N_INITIAL))

        logger.info(f"SIMULATION: Running CDR for betas {np.round(new_betas, 3)}")

        new_pod, new_gp, x_grid_new = _run_cdr_batch(new_betas)

        X_pod_prev = prev.get("X_pod", None)
        X_gp_prev = prev.get("X_gp", None)
        prev_betas = prev.get("all_train_betas", np.array([]))
        x_grid = prev.get("x_grid", x_grid_new)

        X_pod_new = np.concatenate(new_pod, axis=0)
        X_gp_new = np.asarray(new_gp, dtype=float)

        if X_pod_prev is not None:
            X_pod = np.vstack([X_pod_prev, X_pod_new])
            X_gp = np.vstack([X_gp_prev, X_gp_new])
        else:
            X_pod = X_pod_new
            X_gp = X_gp_new

        all_betas = np.concatenate([prev_betas, new_betas])

        return {
            "X_pod": X_pod,
            "X_gp": X_gp,
            "all_train_betas": all_betas,
            "x_grid": x_grid,
        }

    # ================================================================
    # TRAINING TASK
    # ================================================================
    @acl.training_task(as_executable=False)
    async def training(*args, **kwargs) -> dict:
        sim = args[0]
        X_pod = sim["X_pod"]
        X_gp = sim["X_gp"]

        n_comp = min(N_COMPONENTS, X_pod.shape[0])
        pod = POD(n_components=n_comp)
        pod.fit(pd.DataFrame(X_pod))
        coeffs = pod.svd.transform(X_pod)

        surrogate = Surrogate()
        surrogate.train(X_gp, coeffs)

        cand = np.linspace(*BETA_RANGE, CANDIDATE_SIZE)
        X_cand = np.column_stack([cand, np.full(CANDIDATE_SIZE, T_FINAL)])
        _, std = surrogate.predict(X_cand, return_std=True)
        mean_unc = float(np.mean(std)) if std.ndim == 1 else float(np.mean(np.mean(std, axis=1)))

        r2 = surrogate.evaluate(X_gp, coeffs)
        n_betas = len(sim["all_train_betas"])

        logger.info(
            f"TRAINING: {n_betas} betas, {X_pod.shape[0]} snapshots, "
            f"R²={r2:.4f}, mean_uncertainty={mean_unc:.6f}"
        )

        return {
            "pod": pod,
            "surrogate": surrogate,
            "mean_uncertainty": mean_unc,
            "r2_score": r2,
        }

    # ================================================================
    # ACTIVE LEARNING TASK
    # ================================================================
    @acl.active_learn_task(as_executable=False)
    async def active_learn(*args, **kwargs) -> dict:
        sim = args[0]
        train = args[1]

        surrogate = train["surrogate"]

        cand_betas = np.linspace(*BETA_RANGE, CANDIDATE_SIZE)
        X_cand = np.column_stack([cand_betas, np.full(CANDIDATE_SIZE, T_FINAL)])
        _, std = surrogate.predict(X_cand, return_std=True)

        std_scalar = np.mean(std, axis=1) if std.ndim > 1 else std

        top_idx = np.argsort(std_scalar)[-N_PER_ITER:]
        new_betas = np.sort(cand_betas[top_idx])

        logger.info(
            f"ACTIVE LEARN: Selected betas {np.round(new_betas, 3)} "
            f"(max_std={np.max(std_scalar):.6f})"
        )

        return {
            "new_betas": new_betas,
            "X_pod": sim["X_pod"],
            "X_gp": sim["X_gp"],
            "all_train_betas": sim["all_train_betas"],
            "x_grid": sim["x_grid"],
        }

    # ================================================================
    # RUN THE ACTIVE LEARNING LOOP
    # ================================================================
    logger.info("=" * 60)
    logger.info("ROSE Active Learning CDR Surrogate Workflow")
    logger.info("=" * 60)

    history = []
    final_state = None

    async for state in acl.start(max_iter=MAX_ITERATIONS):
        unc = state.mean_uncertainty
        r2 = state.r2_score
        n_betas = len(state.all_train_betas) if state.all_train_betas is not None else 0
        history.append({
            "iteration": state.iteration,
            "mean_uncertainty": unc,
            "r2": r2,
            "n_betas": n_betas,
        })
        logger.info(
            f"==> Iteration {state.iteration}: "
            f"{n_betas} betas, mean_unc={unc:.6f}, R²={r2:.4f}"
        )
        final_state = state

    await acl.shutdown()

    # ================================================================
    # PLOTTING
    # ================================================================
    if final_state is None:
        logger.error("No iterations completed!")
        return

    pod = final_state.pod
    surrogate = final_state.surrogate
    train_betas = final_state.all_train_betas
    x_grid = final_state.x_grid

    test_betas = np.linspace(*BETA_RANGE, N_TEST)

    logger.info("Running ground-truth simulations for comparison...")
    truth_rows = []
    for b in test_betas:
        df, _ = run_cdr(theta=float(b), t_final=T_FINAL, write_every=1000)
        truth_rows.append(df.iloc[:, -1].to_numpy())
    Z_true = np.array(truth_rows)

    X_pred = np.column_stack([test_betas, np.full(N_TEST, T_FINAL)])
    coeffs_pred = surrogate.predict(X_pred)
    Z_surr = pod.svd.inverse_transform(coeffs_pred)

    x_edges = _make_edges(x_grid)
    b_edges = _make_edges(test_betas, lo=BETA_RANGE[0], hi=BETA_RANGE[1])

    vmin = min(Z_true.min(), Z_surr.min())
    vmax = max(Z_true.max(), Z_surr.max())

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.8], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])

    im = ax0.pcolormesh(x_edges, b_edges, Z_true,
                        shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.set_title("Ground Truth")
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$\beta$")

    ax1.pcolormesh(x_edges, b_edges, Z_surr,
                   shading="flat", cmap="viridis", vmin=vmin, vmax=vmax)
    for b in train_betas:
        ax1.axhline(b, color="0.55", linewidth=0.5, linestyle="--", alpha=0.6)
    ax1.set_title(f"ROSE Surrogate ({len(train_betas)} training $\\beta$)")
    ax1.set_xlabel(r"$x$")
    plt.setp(ax1.get_yticklabels(), visible=False)

    fig.colorbar(im, cax=cax, label=r"$u(x,\, t=0.25)$")

    iters = [h["iteration"] for h in history]
    uncs = [h["mean_uncertainty"] for h in history]
    n_bs = [h["n_betas"] for h in history]

    color_unc = "steelblue"
    ax2.plot(iters, uncs, "o-", color=color_unc, linewidth=2, label="Mean uncertainty")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean GP Uncertainty", color=color_unc)
    ax2.tick_params(axis="y", labelcolor=color_unc)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Convergence")

    ax2b = ax2.twinx()
    color_nb = "coral"
    ax2b.plot(iters, n_bs, "s--", color=color_nb, linewidth=1.5, label=r"# training $\beta$")
    ax2b.set_ylabel(r"Training $\beta$ count", color=color_nb)
    ax2b.tick_params(axis="y", labelcolor=color_nb)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    plt.savefig("rose_workflow.png", dpi=200)
    plt.close()
    logger.info("Saved rose_workflow.png")


if __name__ == "__main__":
    asyncio.run(main())
