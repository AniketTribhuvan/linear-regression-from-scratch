# Custom Linear Regression

Linear regression built from scratch using only NumPy, trained with gradient descent.

## Why I built this

I'm a 17 yr old diploma student in AI/ML. I could have just written `LinearRegression().fit(X, y)` from scikit-learn, but I wanted to know what actually happens inside `fit()`. So I wrote the model, the scaling, the train/test split and the metrics myself, with no scikit-learn. I made it behave like a small real library, not just a notebook.

## What I implemented myself

- Gradient descent training loop (`fit`) with hand-derived gradients
- Feature standardization (`scale_train`, `scale_test`)
- Random train/test split with a seed
- MSE, RMSE, MAE and R² metrics
- Input validation and custom exceptions
- Unit tests with pytest (23 tests)

## Features

- **Gradient descent** with MSE loss and a configurable learning rate
- **Three stopping conditions**, plus a `max_iter` limit as a safety net:
  1. **MSE tolerance:** loss is already tiny, so there is nothing left to learn
  2. **Gradient norm:** gradient is almost zero, so we are at the minimum
  3. **Relative MSE change:** loss has barely changed for 5 iterations in a row (training has flattened out)
- **Divergence detection:** stops with a clear error if the loss increases 5 times in a row, or if numbers overflow to infinity (usually a learning rate that is too big)
- **Input validation:** checks types, NaN/inf values, shapes, empty arrays and invalid hyperparameters, with clear error messages
- **No data leakage:** the scaler learns mean and std from the training data only
- **Safe scaling:** features with zero std don't cause division by zero
- **Fitted-model checks:** `predict()` before `fit()` raises `NotFittedError`
- **`stop_reason_`:** after training you can see why it stopped

## The maths

For `n` samples, with predictions `ŷ = Xw + b`:

**Loss (MSE)**

```
L = (1/n) * Σ (ŷ - y)²
```

**Gradients** (with `residuals = ŷ - y`)

```
∂L/∂w = (2/n) * Xᵀ · residuals
∂L/∂b = (2/n) * Σ residuals
```

**Update step** (`α` = learning rate)

```
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b
```

**Gradient norm** (used as a stopping condition)

```
||g|| = sqrt( ||∂L/∂w||² + (∂L/∂b)² )
```

**Standardization:** `z = (x - mean) / std`

**R² score:** `1 - SS_res / SS_tot`

## Project structure

```
linear-regression-from-scratch/

  README.md                       # Project overview, setup instructions, API usage, concepts demonstrated, and documentation

  requirements.txt                # Project dependencies required to run the custom regression library

  main.py                         # Entry point demonstrating model training, prediction, preprocessing, and evaluation

  custom_linear_regression/      # Python package containing all core machine learning implementation
    __init__.py                  # Package initialization and public API exports
    regression.py                # Custom Linear Regression implementation using NumPy and gradient descent
    validation.py                # Input validation for X, y, model parameters, shapes, and finite values
    metrics.py                   # Regression evaluation metrics including MSE, RMSE, MAE, and R²
    preprocessing.py             # Feature scaling and preprocessing utilities
    train_test_split.py          # Train/test dataset splitting with optional shuffling and random state

  tests/                          # Test suite for validating correctness and edge cases
    test_regression.py            # Tests for model initialization, fitting, prediction, convergence, and errors
    test_metrics.py               # Tests for MSE, RMSE, MAE, and R² calculations
    test_preprocessing.py         # Tests for feature scaling and preprocessing behavior

  .gitignore                      # Files and directories excluded from Git tracking

  LICENSE                         # Project license
```

## How to install and run

Needs Python 3.11 or newer.

```bash
pip install -r requirements.txt
python main.py
```

Run the tests:

```bash
python -m pytest
```

## Sample output

`main.py` creates a dataset `y = 3*x1 + 2*x2 - 5*x3 + 4 + noise`, splits it, scales it, trains the model and evaluates it:

```
Train size: 400, Test size: 100
Stopped after 53 iterations. Reason: RELATIVE MSE TOLERANCE
Final training loss (MSE): 25.8405

Test results
MSE      : 24.8200
RMSE     : 4.9820
MAE      : 3.8332
R-square : 0.9991
```

The RMSE of about 5 matches the noise I added (standard deviation 5), so the model has learned the real pattern.

## Usage

```python
from custom_linear_regression.regression import CustomLinearRegression

model = CustomLinearRegression(learning_rate=0.1, max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(model.stop_reason_)
```

## What I learned

- How gradient descent really works and why feature scaling matters for it
- How learning rate affects training, and how to detect when it is too high
- Why stopping conditions are needed and how they differ
- Writing validation, error messages and tests so code fails clearly instead of silently

## License

MIT
