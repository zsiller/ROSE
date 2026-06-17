import numpy as np

import matplotlib.pyplot as plt
from POD import POD
from train import Surrogate

# Spatial and temporal resolution
Nx = 128
Nt = 25
Ns = Nx * Nt  # total degrees of freedom of the field


def run_simulation(beta: float) -> np.ndarray:
    """
    Simple toy "simulation" that returns a field of shape (Nx, Nt)
    for a given parameter beta.
    """
    x = np.linspace(0.0, 1.0, Nx)
    t = np.linspace(0.0, 1.0, Nt)

    # X: spatial coordinate, T: time
    X, T = np.meshgrid(x, t, indexing="ij")

    field = np.sin(beta * X) * np.exp(-T)
    return field


# -------------------------
# Generate training data
# -------------------------
betas_train = np.array([[1.0], [2.0], [3.0]])  # shape (n_samples, 1)

fields_train = []
for beta in betas_train[:, 0]:
    field = run_simulation(beta)
    fields_train.append(field.reshape(1, Ns))  # flatten to a row vector

fields_train = np.concatenate(fields_train, axis=0)  # shape (n_samples, Ns)

print("Training field matrix shape:", fields_train.shape)


# -------------------------
# Build POD basis
# -------------------------
n_components = 3  # reduced order dimension
pod = POD(n_components)
pod.fit(fields_train)

print("Singular values (POD):", pod.svd.singular_values_)

# Project high-dimensional fields into reduced POD space (coefficients)
coeffs_train = pod.svd.transform(fields_train)  # shape (n_samples, n_components)


# -------------------------
# Train Gaussian Process surrogate on POD coefficients
# -------------------------
surrogate = Surrogate()
surrogate.train(betas_train, coeffs_train)

print("Trained surrogate model:", surrogate.model)
print("Training MSE on POD coefficients:", surrogate.evaluate(betas_train, coeffs_train))


# -------------------------
# Test the surrogate at a new parameter and reconstruct the field
# -------------------------
beta_test = np.array([[15]])  # new parameter point
coeffs_pred = surrogate.predict(beta_test)  # predict POD coefficients

# Map predicted coefficients back to full field with the POD model
field_pred_flat = pod.svd.inverse_transform(coeffs_pred)
field_pred = field_pred_flat.reshape(Nx, Nt)

# Plot the predicted field for a quick visual check
plt.figure()
plt.imshow(field_pred, aspect="auto", origin="lower", extent=[0, 1, 0, 1])
plt.colorbar(label="Field value")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title(f"POD-GP surrogate prediction for beta={beta_test[0,0]:.2f}")
plt.tight_layout()

plt.savefig("plot15.png")
plt.close()