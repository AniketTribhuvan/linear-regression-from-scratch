"""Linear Regression built from scratch with NumPy, trained using gradient descent."""

import numpy as np
from typing import Self
from .validation import (
    check_X_y,
    check_X,
    check_arg,
    check_infinite,
    check_is_fitted,
    ShapeMismatchError,
)


class CustomLinearRegression:
    """Linear regression using batch gradient descent and MSE loss.

    Parameters
    ----------
    max_iter : int, maximum number of gradient descent steps.
    grad_tol : float, stop when the gradient norm is <= this value.
    rel_mse_tol : float, stop when the relative change in MSE stays <= this value
        for 5 iterations in a row.
    mse_tol : float, stop when the MSE is <= this value.
    learning_rate : float, step size of each gradient descent update.

    Attributes (available after fit)
    --------------------------------
    coef_ : weights, shape (n_features, 1)
    intercept_ : bias term
    loss_ : final MSE on the training data
    n_iter_ : number of iterations that were run
    stop_reason_ : why training stopped
    """

    def __init__(
        self,
        max_iter=1000,
        grad_tol=1e-6,
        rel_mse_tol=1e-6,
        mse_tol=1e-6,
        learning_rate=0.1,
    ):
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0 or isinstance(max_iter, bool):
            raise ValueError("max_iter must be a positive integer.")
        self.max_iter = max_iter

        self.grad_tol = check_arg(grad_tol, "grad_tol")
        self.rel_mse_tol = check_arg(rel_mse_tol, "rel_mse_tol")
        self.mse_tol = check_arg(mse_tol, "mse_tol")

        if (
            not isinstance(learning_rate, (int, float, np.integer, np.floating))
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number.")
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train the model on X and y using gradient descent."""
        y_is_1d = (np.ndim(y) == 1)
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Start with all weights and bias as zero
        coef = np.zeros((n_features, 1))
        intercept = 0

        divergence_count = 0        # iterations in a row where the loss increased
        rel_mse_change_count = 0    # iterations in a row where the loss barely changed

        stop_reason = None          # reset so calling fit() again works properly

        # Raise an error on overflow / invalid operations instead of silently giving inf or NaN
        try:
            with np.errstate(over="raise", invalid="raise"):

                for i in range(self.max_iter):

                    # 1. Loss at current parameters
                    y_pred = X @ coef + intercept
                    residuals = y_pred - y
                    current_loss = np.dot(residuals.T, residuals).item() / n_samples

                    # 2. Gradients at current parameters
                    grad_coef = (2 / n_samples) * (X.T @ residuals)
                    grad_intercept = (2 / n_samples) * np.sum(residuals)
                    grad_norm = np.sqrt(np.sum(grad_coef ** 2) + grad_intercept ** 2)

                    # 3. Check stopping conditions
                    if current_loss <= self.mse_tol:
                        stop_reason = "MSE CONVERGENCE"
                        break

                    if grad_norm <= self.grad_tol:
                        stop_reason = "GRADIENT NORM"
                        break

                    if i > 0:
                        # Divergence check: loss kept increasing
                        if current_loss > prev_loss:
                            divergence_count += 1
                        else:
                            divergence_count = 0

                        if divergence_count >= 5:
                            raise ValueError(
                                "Training is diverging (loss kept increasing). "
                                "Try a smaller learning_rate or scale your features."
                            )

                        # Relative change check (1e-8 avoids division by zero)
                        rel_mse_change = abs((prev_loss - current_loss) / (prev_loss + 1e-8))
                        if rel_mse_change <= self.rel_mse_tol:
                            rel_mse_change_count += 1
                        else:
                            rel_mse_change_count = 0

                        if rel_mse_change_count >= 5:
                            stop_reason = "RELATIVE MSE TOLERANCE"
                            break

                    # 4. Update parameters
                    coef -= self.learning_rate * grad_coef
                    intercept -= self.learning_rate * grad_intercept
                    prev_loss = current_loss

                else:
                    # Loop finished without breaking, so compute loss at the final parameters
                    stop_reason = "MAX ITERATIONS"
                    y_pred = X @ coef + intercept
                    residuals = y_pred - y
                    current_loss = np.dot(residuals.T, residuals).item() / n_samples

        except FloatingPointError:
            # Any overflow / invalid operation in the loop ends up here
            raise ValueError(
                "Gradients exploded to infinity. "
                "Try a smaller learning_rate or scale your features."
            )

        # Save learned values only after training finished successfully
        self.coef_ = coef
        self.intercept_ = intercept
        self.loss_ = current_loss
        self.n_iter_ = i + 1
        self.stop_reason_ = stop_reason
        self._y_is_1d = y_is_1d   # used in predict() to return the right shape

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
        check_is_fitted(self)

        X = check_X(X)

        if X.shape[1] != self.coef_.shape[0]:
            raise ShapeMismatchError(
                f"X has {X.shape[1]} features but the model was trained with {self.coef_.shape[0]}."
            )

        y_pred = X @ self.coef_ + self.intercept_

        check_infinite(y_pred, "y_pred")

        return y_pred.ravel() if self._y_is_1d else y_pred
