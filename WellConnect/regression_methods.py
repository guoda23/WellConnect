import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


def unconstrained_regression(X, y):
    """
    Standard linear regression without constraints.
    Returns coefficients as np.array.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_


def constrained_regression(X, y):
    """
    Constrained regression: weights >= 0 and sum(weights) = 1.
    Uses scipy.optimize.minimize with SLSQP.
    Returns coefficients as np.array.
    """
    n_features = X.shape[1]

    def loss_fn(w):
        return np.sum((X @ w - y) ** 2)

    constraints = [{'type': 'ineq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, None)] * n_features
    init = np.ones(n_features) / n_features

    result = minimize(
        loss_fn, x0=init, bounds=bounds, constraints=constraints,
        method="SLSQP", options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if not result.success:
        return np.full(n_features, np.nan)
    return result.x


# Function registry
REGRESSION_FUNCTIONS = {
    "unconstrained": unconstrained_regression,
    "constrained": constrained_regression
}