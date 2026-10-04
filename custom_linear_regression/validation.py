"""Input validation helpers and custom exceptions used across the project."""

import numpy as np


class InvalidDataTypeError(Exception):
    """Raised when the input is not a NumPy array / list or is not numeric."""


class ShapeMismatchError(Exception):
    """Raised when two arrays have shapes that don't match."""


class NotFittedError(Exception):
    """Raised when predict() is called before fit()."""


def check_infinite(value, name):
    """Raise ValueError if `value` contains NaN or infinite numbers."""
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values.")


def check_y(y):
    """Validate the target and return it as a 2D array of shape (n_samples, 1)."""
    if not isinstance(y, (np.ndarray, list)):
        raise InvalidDataTypeError(
            f"y must be a NumPy array or list, got {type(y).__name__}."
        )

    if isinstance(y, list):
        y = np.asarray(y)

    if not np.issubdtype(y.dtype, np.number):
        raise InvalidDataTypeError("y must contain only numerical values.")

    check_infinite(y, "y")

    if y.ndim > 2 or y.ndim < 1:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}.")

    if y.ndim == 2 and y.shape[1] != 1:
        raise ValueError(f"y must have exactly 1 column, got shape {y.shape}.")

    if y.shape[0] == 0:
        raise ValueError(f"y has 0 samples (shape={y.shape}), at least 1 is required.")

    # Convert 1D y to a column vector
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return y


def check_X(X):
    """Validate the features and return them as a 2D numeric array."""
    if not isinstance(X, (np.ndarray, list)):
        raise InvalidDataTypeError(
            f"X must be a NumPy array or list, got {type(X).__name__}."
        )

    if isinstance(X, list):
        X = np.asarray(X)

    if not np.issubdtype(X.dtype, np.number):
        raise InvalidDataTypeError("X must contain only numerical values.")

    check_infinite(X, "X")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D array with shape {X.shape}.")

    if X.shape[0] == 0:
        raise ValueError(f"X has 0 samples (shape={X.shape}), at least 1 is required.")

    if X.shape[1] == 0:
        raise ValueError(f"X has 0 features (shape={X.shape}), at least 1 is required.")

    return X


def check_X_y(X, y):
    """Validate X and y together and make sure they have the same number of samples."""
    X = check_X(X)
    y = check_y(y)

    if X.shape[0] != y.shape[0]:
        raise ShapeMismatchError(
            f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}."
        )

    return X, y


def check_arg(value, name):
    """Check that a hyperparameter (like a tolerance) is a finite, non-negative number."""
    if (
        not isinstance(value, (float, int, np.integer, np.floating))
        or not np.all(np.isfinite(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite, non-negative number.")

    return value


def check_is_fitted(instance, attributes=("coef_", "intercept_")):
    """Raise NotFittedError if the model doesn't have its learned attributes yet."""
    if any(not hasattr(instance, attr) or getattr(instance, attr) is None for attr in attributes):
        raise NotFittedError(
            f"This {type(instance).__name__} instance is not fitted yet. "
            "Call 'fit' before using it."
        )


def check_parameter(parameter, X_features):
    """Validate a per-feature parameter (like mean or std) and return it as a 1D array."""
    if not isinstance(parameter, (np.ndarray, list)):
        raise TypeError(f"parameter must be a NumPy array or list, got {type(parameter).__name__}.")

    if isinstance(parameter, list):
        parameter = np.asarray(parameter)

    if not np.issubdtype(parameter.dtype, np.number):
        raise InvalidDataTypeError("parameter must contain only numerical values.")

    if parameter.ndim != 1:
        if parameter.ndim == 2 and parameter.shape[1] == 1:
            parameter = parameter.ravel()
        else:
            raise ValueError("parameter must be 1D or 2D with exactly 1 column.")

    if len(parameter) != X_features:
        raise ShapeMismatchError(
            f"parameter has {len(parameter)} values but X has {X_features} features."
        )

    check_infinite(parameter, "parameter")

    return parameter
