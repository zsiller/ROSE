import matplotlib.pyplot as plt
import numpy as np

uncertainty_mse = [
    0.10946596698397548,
    0.06903100777387956,
    0.04604274372191492,
    0.039716700403624225,
    0.007258323369823505,
    0.007795257159475089,
    0.005756796230454858,
    0.005629382283466939,
    0.003227106866828539,
    0.0023137072559380714,
]

random_mse = [
    0.10706861388331382,
    0.11637749427824041,
    0.06321339191083625,
    0.026631426441981357,
    0.02025258811917915,
    0.017757416072431982,
    0.012776666245268393,
    0.012647640611352626,
    0.009180719448898813,
    0.00788124023208814,
]

iterations = np.arange(len(uncertainty_mse))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iterations, uncertainty_mse, "o-", linewidth=2, label="Uncertainty Sampling")
ax.plot(iterations, random_mse, "s--", linewidth=2, label="Random Sampling")
ax.set_xlabel("Iteration")
ax.set_ylabel("RMSE")
ax.set_title("Active Learning: Uncertainty vs Random Sampling")
    
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/mse_comparison.png", dpi=200)
plt.close()
print("Saved figures/mse_comparison.png")
