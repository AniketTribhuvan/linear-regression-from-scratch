"""Regression evaluation metrics: MSE, RMSE, MAE and R-squared."""

import numpy as np
from .validation import check_y, ShapeMismatchError


def _check_y_pair(y_true, y_pred):
    """Validate y_true and y_pred and make sure their shapes match."""
    y_true, y_pred = check_y(y_true), check_y(y_pred)

    if y_true.shape != y_pred.shape:
        raise ShapeMismatchError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )

    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred contain 0 samples.")

    return y_true, y_pred


def mse(y_true, y_pred):
    """Mean Squared Error."""
    # Raise an error on overflow / invalid operations instead of silently giving inf or NaN
    try:
        with np.errstate(over="raise", invalid="raise"):
            y_true, y_pred = _check_y_pair(y_true, y_pred)

            n_samples = y_true.shape[0]
            residuals = y_true - y_pred
            return np.dot(residuals.T, residuals).item() / n_samples

    except FloatingPointError:
        raise ValueError("Overflow or invalid operation while calculating MSE.")


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            y_true, y_pred = _check_y_pair(y_true, y_pred)

            residuals = y_true - y_pred
            return np.mean(np.abs(residuals)).item()

    except FloatingPointError:
        raise ValueError("Overflow or invalid operation while calculating MAE.")


def r_square(y_true, y_pred):
    """R-squared score: 1 - (residual sum of squares / total sum of squares)."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            y_true, y_pred = _check_y_pair(y_true, y_pred)

            residuals = y_true - y_pred
            ss_res = np.dot(residuals.T, residuals).item()

            deviations = y_true - np.mean(y_true)
            ss_tot = np.dot(deviations.T, deviations).item()

            if ss_tot == 0:
                raise ValueError("Total sum of squares is zero, so R-squared is undefined.")
            if ss_res == 0:
                return 1

            return 1 - (ss_res / ss_tot)

    except FloatingPointError:
        raise ValueError("Overflow or invalid operation while calculating R-squared.")
