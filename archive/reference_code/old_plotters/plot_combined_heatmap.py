import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from simulations.CDR_1D.sim_wrapper import run_cdr

T_FINAL = 0.25
N_BETAS = 50
BETA_RANGE = (0.0, 10.0)


def get_ground_truth(betas, x_grid=None):
    solutions = []
    for beta in betas:
        df, _ = run_cdr(theta=float(beta), t_final=T_FINAL, write_every=1000)
        solutions.append(df.iloc[:, -1].to_numpy())
        if x_grid is None:
            x_grid = df.index.to_numpy()
    return np.array(solutions), x_grid


def get_surrogate_prediction(betas, surrogate_file, pod_file):
    with open(surrogate_file, "rb") as f:
        surrogate = pickle.load(f)
    with open(pod_file, "rb") as f:
        pod = pickle.load(f)
    X_query = np.column_stack([betas, np.full(len(betas), T_FINAL)])
    coeffs = surrogate.predict(X_query)
    return pod.svd.inverse_transform(coeffs)


def make_edges(centers, lo=None, hi=None):
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


if __name__ == "__main__":
    betas = np.linspace(*BETA_RANGE, N_BETAS)

    print("Computing ground truth...")
    Z_true, x_grid = get_ground_truth(betas)

    print("Loading uncertainty surrogate...")
    Z_unc = get_surrogate_prediction(betas, "uncertainty_surrogate.pkl", "uncertainty_pod.pkl")
    with open("uncertainty_beta_history.json", "r") as f:
        unc_betas = json.load(f)

    print("Loading random surrogate...")
    Z_rnd = get_surrogate_prediction(betas, "random_surrogate.pkl", "random_pod.pkl")
    with open("random_beta_history.json", "r") as f:
        rnd_betas = json.load(f)

    x_edges = make_edges(x_grid)
    b_edges = make_edges(betas, lo=BETA_RANGE[0], hi=BETA_RANGE[1])

    vmin = min(Z_true.min(), Z_unc.min(), Z_rnd.min())
    vmax = max(Z_true.max(), Z_unc.max(), Z_rnd.max())

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    panels = [
        (axes[0], Z_true, "Ground Truth", None),
        (axes[1], Z_unc, "Uncertainty Sampling", unc_betas),
        (axes[2], Z_rnd, "Random Sampling", rnd_betas),
    ]

    for ax, Z, title, beta_hist in panels:
        im = ax.pcolormesh(x_edges, b_edges, Z, shading="flat",
                           cmap="viridis", vmin=vmin, vmax=vmax)
        if beta_hist is not None:
            for b in beta_hist:
                ax.axhline(b, color="white", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_xlabel(r"$x$")
        ax.set_title(title)

    axes[0].set_ylabel(r"$\beta$")

    fig.colorbar(im, ax=axes, label=r"$u(x,\, t=0.25)$", shrink=1)

    plt.savefig("figures/combined_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/combined_heatmap.png")
