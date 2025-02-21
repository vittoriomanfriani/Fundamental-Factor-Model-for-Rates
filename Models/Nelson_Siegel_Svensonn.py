import numpy as np
from scipy.optimize import minimize


def nelson_siegel_svensson(params, maturities):

    beta0, beta1, beta2, beta3, lambd1, lambd2 = params

    # Safeguard lambdas to prevent division by zero
    lambd1 = max(lambd1, 1e-4)
    lambd2 = max(lambd2, 1e-4)

    # Safeguard maturities to avoid division by zero
    t = np.where(maturities == 0, 1e-6, maturities)

    alpha_1 = (1 - np.exp(-t / lambd1)) / (t / lambd1)
    alpha_2 = alpha_1 - np.exp(-t / lambd1)
    alpha_3 = (1 - np.exp(-t / lambd2)) / (t / lambd2) - np.exp(-t / lambd2)

    return beta0 + beta1 * alpha_1 + beta2 * alpha_2 + beta3 * alpha_3


def error_function(params, maturities, data):
    data_hat = nelson_siegel_svensson(params, maturities)
    return np.sum((data - data_hat) ** 2)


def ridge_error_function(params, maturities, data, alpha=0.1):
    data_hat = nelson_siegel_svensson(params, maturities)
    error = np.sum((data - data_hat) ** 2)
    regularization = alpha * sum(param**2 for param in params)
    return error + regularization


def fit_nelson_siegel_svensson(maturities, yields, ridge=False, alpha=0.1, initial_params=None):
    """
    Fit the Nelson-Siegel-Svensson model to yield curve data.

    Parameters:
    - maturities: List or array of maturities (e.g., in years).
    - yields: List or array of corresponding yield values.
    - ridge: Whether to use ridge regularization.
    - alpha: Ridge regularization parameter.
    - initial_params: Initial guess for the parameters [beta0, beta1, beta2, beta3, lambda1, lambda2].
    - lambda_bounds: Tuple specifying the lower and upper bounds for lambda1 and lambda2.

    Returns:
    - Optimized parameters as a numpy array.
    """
    if initial_params is None:
        initial_params = [3, 0, 0, 0, 1, 1]
    lambda_bounds = (0.5, 3)
    bounds = [
        (-np.inf, np.inf),  # No bounds for beta0
        (-np.inf, np.inf),  # No bounds for beta1
        (-np.inf, np.inf),  # No bounds for beta2
        (-np.inf, np.inf),  # No bounds for beta3
        lambda_bounds,      # Bounds for lambda1
        lambda_bounds       # Bounds for lambda2
    ]
    if ridge:
        result = minimize(
            ridge_error_function,
            initial_params,
            args=(maturities, yields, alpha),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )
    else:
        result = minimize(
            error_function,
            initial_params,
            args=(maturities, yields),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

    return result.x