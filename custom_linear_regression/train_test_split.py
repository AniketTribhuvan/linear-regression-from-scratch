"""Function to split a dataset into train and test sets."""

import numpy as np
from .validation import check_X_y


def train_test_split(X, y, test_ratio=0.2, random_state=20):
    """Randomly split X and y into train and test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,) or (n_samples, 1)
    test_ratio : float, between 0 and 1 (exclusive). Fraction used for testing.
    random_state : int or None. Seed for shuffling, use the same value to get the same split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y_is_1d = (np.ndim(y) == 1)     # remember this before check_X_y makes y 2D
    X, y = check_X_y(X, y)
    n_samples = X.shape[0]

    # Validate test_ratio
    if not isinstance(test_ratio, (int, float)) or isinstance(test_ratio, bool):
        raise TypeError("test_ratio must be an int or float (not bool).")
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive).")

    # Validate random_state
    if random_state is not None and (
        not isinstance(random_state, int) or isinstance(random_state, bool)
    ):
        raise TypeError("random_state must be an int or None.")

    # Number of samples that go into the test set
    test_samples = round(n_samples * test_ratio)
    if test_samples <= 0 or test_samples >= n_samples:
        raise ValueError(
            f"This split leaves 0 samples in the {'test' if test_samples <= 0 else 'train'} set. "
            "Use a bigger dataset or change test_ratio."
        )

    # Shuffle the row indices, then take the first part as test
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Give y back in the same shape it came in
    if y_is_1d:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test
