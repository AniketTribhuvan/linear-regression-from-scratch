import numpy as np
import pytest

from custom_linear_regression.preprocessing import scale_train, scale_test
from custom_linear_regression.validation import ShapeMismatchError

X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


def test_scale_train_mean_is_zero_and_std_is_one():
    scaled_X, mean, std = scale_train(X_train)

    assert np.allclose(scaled_X.mean(axis=0), 0)
    assert np.allclose(scaled_X.std(axis=0), 1)
    assert np.allclose(mean, [2, 20])


def test_scale_train_constant_column_has_no_nan():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaled_X, mean, std = scale_train(X)

    assert not np.isnan(scaled_X).any()
    assert std[1] == 1      # std = 0 is replaced with 1


def test_scale_test_uses_train_mean_and_std():
    _, mean, std = scale_train(X_train)
    X_test = np.array([[2.0, 20.0]])
    scaled_test = scale_test(X_test, mean, std)

    # test row is equal to the train mean, so it should become 0
    assert np.allclose(scaled_test, 0)


def test_scale_test_wrong_mean_length_raises_error():
    X_test = np.array([[2.0, 20.0]])
    with pytest.raises(ShapeMismatchError):
        scale_test(X_test, [2.0], [1.0, 1.0])


def test_scale_train_non_numeric_raises_error():
    with pytest.raises(Exception):
        scale_train(np.array([["a", "b"], ["c", "d"]]))
