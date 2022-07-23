import numpy as np
from scipy.stats import norm

# Functional Structure

def probability_improvement(X: np.ndarray, X_sample: np.ndarray,
                            gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Probability improvement acquisition function.

    Computes the PI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        PI: ndarray of shape (m,)
    """
    # TODO Q2.4
    # Implement the probability of improvement acquisition function
    m = X.shape[0]
    # X sample is used to perform the f function.
    y_sample_mean = gpr.predict(X_sample,return_std = False)
    # use X to obtain y_mean and y_std
    y_mean, y_std = gpr.predict(X,return_std = True)
    fmax = np.max(y_sample_mean)
    Z = (y_mean - fmax - xi )/ y_std
    return norm.cdf(Z).reshape(m,)


def expected_improvement(X: np.ndarray, X_sample: np.ndarray,
                         gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Computes the EI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        EI : ndarray of shape (m,)
    """

    # TODO Q2.4
    # Implement the expected improvement acquisition function
    # X sample is used to perform the f function.
    y_sample_mean = gpr.predict(X_sample,return_std = False)
    # use X to obtain y_mean and y_std
    y_mean, y_std = gpr.predict(X,return_std = True)
    fmax = np.max(y_sample_mean)
    m = X.shape[0]
    EI = np.zeros(m,)
    for idx, std in enumerate(y_std):
        if std != 0:
            a = (y_mean[idx] - fmax - xi)
            z = a / std
            EI[idx] = (a * norm.cdf(z) + std * norm.pdf(z))
    return EI
