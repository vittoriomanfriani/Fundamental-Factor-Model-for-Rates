import numpy as np
from scipy.optimize import minimize

def nelson_siegel(params, maturities):
    beta0, beta1, beta2, lambd = params

    # set a min value for lambda to account for 0 division in optimization problems
    lambd = max(lambd, 1e-4)


    # Safeguard maturities to avoid division by zero
    t = np.where(maturities == 0, 1e-6, maturities)

    alpha_1 = (1 - np.exp(-t/lambd))/(t/lambd)
    alpha_2 = (1 - np.exp(-t/lambd))/(t/lambd) - np.exp(-t/lambd)
    return beta0 + beta1 * alpha_1 + beta2 * alpha_2

# Error function to minimize to find optimal params
def error_function(params, maturities, data):
    data_hat = nelson_siegel(params, maturities)
    return np.sum((data - data_hat) ** 2)

# We define ridge error function as
def ridge_error_function(params, maturities, data, alpha=0.1):
    data_hat = nelson_siegel(params, maturities)
    error = np.sum((data - data_hat) ** 2)
    regularization = alpha * (params[0]**2 + params[1]**2 + params[2]**2 + params[3]**2)
    return error + regularization


def fit_nelson_siegel(maturities, yields, ridge=False, alpha=0.1, initial_params=None):
    """
    Fit the Nelson-Siegel model to yield curve data.

    Parameters:
    - maturities: List or array of maturities (e.g., in years).
    - yields: List or array of corresponding yield values.
    - ridge: Whether to use ridge regularization.
    - alpha: Ridge regularization parameter.
    - initial_params: Initial guess for the parameters [beta0, beta1, beta2, lambda].
    - lambda_bounds: Tuple specifying the lower and upper bounds for lambda.

    Returns:
    - Optimized parameters as a numpy array.
    """

    if initial_params is None:
        initial_params = [3, 0, 0, 0.5]


    # Define bounds: (min, max) for each parameter
    bounds = [
        (-np.inf, np.inf),  # No bounds for beta0
        (-np.inf, np.inf),  # No bounds for beta1
        (-np.inf, np.inf),  # No bounds for beta2
        (0.5, 3)       # Bounds for lambda
    ]

    if ridge:
        result = minimize(
            ridge_error_function,
            initial_params,
            args=(maturities, yields, alpha),
            method="L-BFGS-B",
            bounds = bounds,
            options={"maxiter": 1000},
        )
    else:
        result = minimize(
            error_function,
            initial_params,
            args=(maturities, yields),
            method="L-BFGS-B",
            bounds = bounds,
            options={"maxiter": 1000},
        )

    return result.x