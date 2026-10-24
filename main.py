"""Runs the whole Custom Linear Regression project from start to end."""

import numpy as np

from custom_linear_regression.regression import CustomLinearRegression
from custom_linear_regression.preprocessing import scale_train, scale_test
from custom_linear_regression.train_test_split import train_test_split
from custom_linear_regression.metrics import mse, rmse, mae, r_square


def main():
    # 1. Create a simple dataset: y = 3*x1 + 2*x2 - 5*x3 + 4 + noise
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, size=(500, 3))
    true_coef = np.array([3, 2, -5])
    true_intercept = 4
    noise = rng.normal(0, 5, size=500)
    y = X @ true_coef + true_intercept + noise

    # 2. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, random_state=20)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. Scale features (mean and std come from train data only)
    X_train_scaled, mean, std = scale_train(X_train)
    X_test_scaled = scale_test(X_test, mean, std)

    # 4. Train the model
    model = CustomLinearRegression(max_iter=1000, learning_rate=0.1)
    model.fit(X_train_scaled, y_train)
    print(f"Stopped after {model.n_iter_} iterations. Reason: {model.stop_reason_}")
    print(f"Final training loss (MSE): {model.loss_:.4f}")

    # 5. Predict on test data
    y_pred = model.predict(X_test_scaled)

    # 6. Evaluate
    print("\nTest results")
    print(f"MSE      : {mse(y_test, y_pred):.4f}")
    print(f"RMSE     : {rmse(y_test, y_pred):.4f}")
    print(f"MAE      : {mae(y_test, y_pred):.4f}")
    print(f"R-square : {r_square(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
