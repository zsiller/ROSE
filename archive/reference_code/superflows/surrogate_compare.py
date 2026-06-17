import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulations.CDR_1D.sim_wrapper import run_cdr
from surrogate.POD import POD
from surrogate.train import Surrogate


def build_surrogate(train_betas, t_final=0.25, n_components=5):
    """Run CDR for each training beta, fit POD + GP, return (pod, surrogate)."""
    X_pod_rows = []
    X_gp_rows = []

    for beta in train_betas:
        df, _ = run_cdr(theta=beta, t_final=t_final, write_every=1000)
        times = df.columns.to_numpy()
        for t in times:
            X_pod_rows.append(df[t].to_numpy().reshape(1, -1))
            X_gp_rows.append([beta, float(t)])

    X_pod = np.concatenate(X_pod_rows, axis=0)
    X_gp = np.asarray(X_gp_rows, dtype=float)

    n_comp = min(n_components, X_pod.shape[0])
    pod = POD(n_components=n_comp)
    pod.fit(pd.DataFrame(X_pod))
    coeffs = pod.svd.transform(X_pod)

    surrogate = Surrogate()
    surrogate.train(X_gp, coeffs)
    return pod, surrogate


def predict_heatmap(pod, surrogate, test_betas, t_query=0.25):
    """Use the surrogate to predict u(x, t_query) for each test beta."""
    X_test = np.column_stack([test_betas, np.full_like(test_betas, t_query)])
    coeffs_pred = surrogate.predict(X_test)
    return pod.svd.inverse_transform(coeffs_pred)


def truth_heatmap(test_betas, t_final=0.25):
    """Run the actual CDR solver for each test beta, return (Z_true, x_grid)."""
    rows = []
    x_grid = None
    for beta in test_betas:
        df, _ = run_cdr(theta=beta, t_final=t_final, write_every=1000)
        rows.append(df.iloc[:, -1].to_numpy())
        if x_grid is None:
            x_grid = df.index.to_numpy()
    return np.array(rows), x_grid


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
    rng = np.random.default_rng(42)

    n_test = 50
    test_betas = np.linspace(0, 10, n_test)
    train_counts = [2, 10]

    print("=== Running ground-truth simulations ===")
    Z_true, x_grid = truth_heatmap(test_betas)

    surrogates = {}
    for n_train in train_counts:
        train_betas = np.sort(rng.uniform(0, 10, size=n_train))
        print(f"\n=== Training surrogate with {n_train} betas: "
              f"{np.round(train_betas, 3)} ===")
        pod, surr = build_surrogate(train_betas)
        Z_pred = predict_heatmap(pod, surr, test_betas)
        surrogates[n_train] = {"Z": Z_pred, "train_betas": train_betas}

    x_edges = make_edges(x_grid)
    b_edges = make_edges(test_betas, lo=0.0, hi=10.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    vmin = min(Z_true.min(),
               *(s["Z"].min() for s in surrogates.values()))
    vmax = max(Z_true.max(),
               *(s["Z"].max() for s in surrogates.values()))

    im = axes[0].pcolormesh(x_edges, b_edges, Z_true,
                            shading="flat", cmap="viridis",
                            vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth (50 sims)")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$\beta$")

    for ax, n_train in zip(axes[1:], train_counts):
        info = surrogates[n_train]
        ax.pcolormesh(x_edges, b_edges, info["Z"],
                      shading="flat", cmap="viridis",
                      vmin=vmin, vmax=vmax)
        for b in info["train_betas"]:
            ax.axhline(b, color="1", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_title(f"Surrogate ({n_train} training $\\beta$)")
        ax.set_xlabel(r"$x$")

    fig.subplots_adjust(left=0.05, right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r"$u(x,\, t=0.25)$")
    plt.savefig("surrogate_compare.png", dpi=200)
    plt.close()
    print("\nSaved surrogate_compare.png")
