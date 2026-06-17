import numpy as np
import matplotlib.pyplot as plt
from simulations.CDR_1D.sim_wrapper import run_cdr

if __name__ == "__main__":
    betas = np.linspace(0, 10, 50)

    solutions = []
    x_grid = None

    for i, beta in enumerate(betas):
        print(f"Running beta={beta:.4f}  ({i + 1}/{len(betas)})")
        df, meta = run_cdr(theta=beta, t_final=0.25, write_every=1000)

        state = df.iloc[:, -1].to_numpy()
        solutions.append(state)

        if x_grid is None:
            x_grid = df.index.to_numpy()

    Z = np.array(solutions)

    x_edges = np.concatenate([[x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])],
                               0.5 * (x_grid[:-1] + x_grid[1:]),
                               [x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])]])
    b_edges = np.concatenate([[betas[0] - 0.5 * (betas[1] - betas[0])],
                               0.5 * (betas[:-1] + betas[1:]),
                               [betas[-1] + 0.5 * (betas[-1] - betas[-2])]])
    b_edges[0] = max(b_edges[0], 0.0)
    b_edges[-1] = min(b_edges[-1], 10.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x_edges, b_edges, Z, shading="flat", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$u(x,\, t=0.25)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"CDR Solution at $t = 0.25$ for varying $\beta$")
    plt.tight_layout()
    plt.savefig("cdr_heatmap.png", dpi=200)
    plt.close()
