import numpy as np
import pytest

from custom_linear_regression.metrics import mse, rmse, mae, r_square
from custom_linear_regression.validation import ShapeMismatchError

y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 5])


def test_mse():
    assert mse(y_true, y_pred) == pytest.approx(4 / 3)


def test_rmse():
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))


def test_mae():
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)


def test_r_square():
    # ss_res = 4, ss_tot = 2, so r2 = 1 - 4/2 = -1
    assert r_square(y_true, y_pred) == pytest.approx(-1)


def test_perfect_prediction():
    assert mse(y_true, y_true) == 0
    assert r_square(y_true, y_true) == 1


def test_shape_mismatch_raises_error():
    with pytest.raises(ShapeMismatchError):
        mse([1, 2, 3], [1, 2])


def test_r_square_constant_y_raises_error():
    with pytest.raises(ValueError):
        r_square([2, 2, 2], [1, 2, 3])


def test_nan_input_raises_error():
    with pytest.raises(ValueError):
        mae([1, 2, np.nan], [1, 2, 3])
