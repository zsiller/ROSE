import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from simulations.CDR_1D.sim_wrapper import run_cdr


if __name__ == "__main__":
    with open("uncertainty_surrogate.pkl", "rb") as f:
        surrogate = pickle.load(f)
    with open("random_pod.pkl", "rb") as f:
        pod = pickle.load(f)

    betas = np.linspace(0, 10, 50)
    T_FINAL = 0.25

    X_query = np.column_stack([betas, np.full(len(betas), T_FINAL)])
    coeffs_pred = surrogate.predict(X_query)
    Z_surr = pod.svd.inverse_transform(coeffs_pred)

    # Get x_grid from a single simulation
    df, _ = run_cdr(theta=1.0, t_final=T_FINAL, write_every=1000)
    x_grid = df.index.to_numpy()

    x_edges = np.concatenate([[x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])],
                               0.5 * (x_grid[:-1] + x_grid[1:]),
                               [x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])]])
    b_edges = np.concatenate([[betas[0] - 0.5 * (betas[1] - betas[0])],
                               0.5 * (betas[:-1] + betas[1:]),
                               [betas[-1] + 0.5 * (betas[-1] - betas[-2])]])
    b_edges[0] = max(b_edges[0], 0.0)
    b_edges[-1] = min(b_edges[-1], 10.0)

    with open("uncertainty_beta_history.json", "r") as f:
        beta_history = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x_edges, b_edges, Z_surr, shading="flat", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$u(x,\, t=0.25)$")
    for b in beta_history:
        ax.axhline(b, color="white", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Uncertainty Surrogate Prediction at $t = 0.25$ for varying $\beta$")
    plt.tight_layout()
    plt.savefig("figures/uncertainty_heatmap.png", dpi=200)
    plt.close()
    print("Saved figures/uncertainty_heatmap.png")
