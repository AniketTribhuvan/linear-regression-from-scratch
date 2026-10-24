import numpy as np
import pytest

from custom_linear_regression.regression import CustomLinearRegression
from custom_linear_regression.validation import NotFittedError, ShapeMismatchError


def make_data():
    """Perfect linear data: y = 3*x1 - 2*x2 + 5 (no noise)."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5
    return X, y


def test_fit_learns_correct_values():
    X, y = make_data()
    model = CustomLinearRegression(max_iter=5000).fit(X, y)

    assert model.coef_.ravel() == pytest.approx([3, -2], abs=0.01)
    assert model.intercept_ == pytest.approx(5, abs=0.01)


def test_predict_matches_y():
    X, y = make_data()
    model = CustomLinearRegression(max_iter=5000).fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.ravel() == pytest.approx(y, abs=0.05)


def test_fit_sets_attributes():
    X, y = make_data()
    model = CustomLinearRegression().fit(X, y)

    assert model.n_iter_ > 0
    assert model.stop_reason_ is not None
    assert model.loss_ >= 0


def test_fit_twice_works():
    X, y = make_data()
    model = CustomLinearRegression()
    model.fit(X, y)
    first_coef = model.coef_.copy()
    model.fit(X, y)

    assert np.allclose(first_coef, model.coef_)


def test_predict_before_fit_raises_error():
    model = CustomLinearRegression()
    with pytest.raises(NotFittedError):
        model.predict(np.array([[1.0, 2.0]]))


def test_predict_wrong_number_of_features_raises_error():
    X, y = make_data()
    model = CustomLinearRegression().fit(X, y)
    with pytest.raises(ShapeMismatchError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_x_y_sample_mismatch_raises_error():
    X, y = make_data()
    with pytest.raises(ShapeMismatchError):
        CustomLinearRegression().fit(X, y[:10])


def test_invalid_learning_rate_raises_error():
    with pytest.raises(ValueError):
        CustomLinearRegression(learning_rate=-0.1)


def test_invalid_max_iter_raises_error():
    with pytest.raises(ValueError):
        CustomLinearRegression(max_iter=0)


def test_divergence_raises_error():
    X, y = make_data()
    model = CustomLinearRegression(learning_rate=100)
    with pytest.raises(ValueError):
        model.fit(X, y)
