import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from helpers.log import get_logger
from task_train.POD import POD

logger = get_logger(__name__)

DEFAULT_KERNEL = 1.0 * Matern(
    length_scale=np.ones(5), length_scale_bounds=(1e-3, 1e3), nu=1.5
) + 1.0 * WhiteKernel(noise_level=1.0, )


class Surrogate:
    """GP on POD coefficients with StandardScaler on inputs and solution fields."""

    def __init__(self, kernel=None, n_pod_components: int = 20, pod_inc: bool = False):
        self.kernel = kernel if kernel is not None else DEFAULT_KERNEL
        self.n_pod_components = n_pod_components
        self.pod_inc = pod_inc
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=10
        )
        self.pod = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.n_trainings = 0

    def train(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        reoptimize: bool = True,
    ) -> None:
        """Fit scalers, POD on scaled fields, GP on scaled X -> POD coeffs.

        Always fits on the full data passed in (a GP "remembers" only the data in
        its fit). When ``reoptimize`` is False and the GP was already fitted, this
        performs a warm restart: it reuses the previously learned kernel
        hyperparameters and skips the expensive marginal-likelihood optimization,
        doing only a single factorization over all rows.
        """
        X_scaled = self.x_scaler.fit_transform(X_labeled)
        Y_scaled = self.y_scaler.fit_transform(y_labeled)

        already_fitted = hasattr(self.gp, "X_train_")
        # The marginal-likelihood optimizer runs on the first fit and on every
        # reoptimize, but NOT on a frozen warm restart (optimizer=None).
        optimized = (not already_fitted) or reoptimize
        if already_fitted:
            if reoptimize:
                # Re-optimize hyperparameters, warm-starting the search from the
                # previously learned kernel. The GP may currently have its
                # optimizer disabled (from a prior warm restart), so we must
                # rebuild it with the optimizer re-enabled or fit() would be a
                # no-op on the hyperparameters.
                self.gp = GaussianProcessRegressor(
                    kernel=self.gp.kernel_,
                    n_restarts_optimizer=10,
                )
                logger.info(
                    "Reoptimizing hyperparameters from %s", self.gp.kernel
                )
            else:
                # Warm restart: freeze the learned hyperparameters and skip the
                # marginal-likelihood optimization, doing only a single
                # factorization over all rows.
                self.gp = GaussianProcessRegressor(
                    kernel=self.gp.kernel_,
                    optimizer=None,
                    n_restarts_optimizer=0,
                )
                logger.info("Warm restart: frozen hyperparameters %s", self.gp.kernel)

        if self.pod_inc:
            n_comp = min(
                self.n_pod_components,
                X_labeled.shape[0],
                y_labeled.shape[1],
            )
            self.pod = POD(n_components=n_comp)
            self.pod.fit(Y_scaled)
            coeffs = self.pod.svd.transform(Y_scaled)
            self.gp.fit(X_scaled, coeffs)
            logger.info("Using POD with %d components", n_comp)
        else:
            self.gp.fit(X_scaled, Y_scaled)

        # Report the kernel hyperparameters the optimizer actually landed on
        # (kernel_ is the fitted kernel; only meaningful when the optimizer ran).
        if optimized:
            logger.info("Optimized kernel hyperparameters: %s", self.gp.kernel_)

        self.n_trainings += 1

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        """Predict POD coefficients from physical inputs [p_l, p_r, rho_l, rho_r, t]."""
        X_scaled = self.x_scaler.transform(np.atleast_2d(X_test))
        std_scalar = None
        if self.pod_inc:
            if return_std:
                coeffs, std = self.gp.predict(X_scaled, return_std=True)
                Y = self.pod.svd.inverse_transform(coeffs)
                std = self.pod.svd.inverse_transform(std)
                std_scalar = np.mean(std, axis=1) if std.ndim > 1 else std
            else:
                coeffs = self.gp.predict(X_scaled, return_std=False)
                Y = self.pod.svd.inverse_transform(coeffs)
        else:
            if return_std:
                Y, std = self.gp.predict(X_scaled, return_std=True)
                std_scalar = np.mean(std, axis=1) if std.ndim > 1 else std
            else:
                Y = self.gp.predict(X_scaled, return_std=False)
        return self.y_scaler.inverse_transform(Y), std_scalar


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """R^2 score in POD coefficient space."""
        if self.pod_inc:
            raise RuntimeError("Surrogate must be trained before evaluate")
        Y_scaled = self.y_scaler.transform(y_test)
        coeffs = self.pod.svd.transform(Y_scaled)
        X_scaled = self.x_scaler.transform(np.atleast_2d(X_test))
        return float(self.gp.score(X_scaled, coeffs))
