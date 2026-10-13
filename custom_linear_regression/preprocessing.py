"""Feature scaling (standardization) helpers."""

import numpy as np
from .validation import check_X, check_parameter


def scale_train(X):
    """Standardize training data: (X - mean) / std.

    Returns the scaled X along with the mean and std, so the same
    values can be reused on the test data with scale_test().
    """
    X = check_X(X)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Features with std = 0 are divided by 1 to avoid division by zero
    safe_std = np.where(std == 0, 1, std)

    scaled_X = (X - mean) / safe_std

    return scaled_X, mean, safe_std


def scale_test(X, mean, std):
    """Standardize test data using the mean and std learned from the training data."""
    X = check_X(X)

    n_features = X.shape[1]

    mean = check_parameter(mean, n_features)
    std = check_parameter(std, n_features)

    safe_std = np.where(std == 0, 1, std)

    scaled_X = (X - mean) / safe_std

    return scaled_X
