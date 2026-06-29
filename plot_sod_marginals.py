"""Plot the 4 Sod IC parameter marginals from sod_chain_full.dat with truth lines."""
import numpy as np
import matplotlib.pyplot as plt

DATA = "sod_chain_full.dat"
PARAMS = ["p_high", "p_low", "rho_high", "rho_low"]
TRUTH = {"p_high": 1e5, "p_low": 1e4, "rho_high": 1.0, "rho_low": 0.125}

data = np.genfromtxt(DATA, names=True)
# keep posterior-phase samples (phase == 1), drop burn-in
mask = data["phase"] == 1
samples = data[mask] if mask.any() else data

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, name in zip(axes.ravel(), PARAMS):
    ax.hist(samples[name], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(TRUTH[name], color="red", lw=2, label=f"truth = {TRUTH[name]:g}")
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.legend()

fig.suptitle("Sod IC posterior marginals")
fig.tight_layout()
out = "sod_chain_marginals.png"
fig.savefig(out, dpi=150)
print(f"saved {out} ({len(samples)} samples)")
