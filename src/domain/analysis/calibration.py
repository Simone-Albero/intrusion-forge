import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def fit_platt(score: np.ndarray, rate: np.ndarray) -> tuple[float, float]:
    """Intercept and non-negative slope of `expit(a + b * score)`, fitted to `rate`."""
    # Cross-entropy on each region's rate, every region weighted alike: the rho and the
    # MSE the calibrated baseline is judged by weigh regions alike too, and row weights
    # let a few large regions turn the slope against the ranking of all the others.
    score = np.asarray(score, dtype=float)
    rate = np.asarray(rate, dtype=float)
    pooled = rate.mean()
    mean, sd = score.mean(), score.std()
    if sd == 0.0:
        return float(logit(pooled)), 0.0
    # The optimiser's gradient tolerance is absolute: on a score spanning 1e-3 the slope's
    # gradient starts below it and the fit would stop where it began.
    unit = (score - mean) / sd

    def loss(params: np.ndarray) -> tuple[float, np.ndarray]:
        z = params[0] + params[1] * unit
        residual = expit(z) - rate
        value = np.mean(np.logaddexp(0.0, z) - rate * z)
        return value, np.array([residual.mean(), (residual * unit).mean()])

    # A negative slope would reverse the ranking a calibration has to keep.
    fit = minimize(
        loss,
        x0=np.array([logit(pooled), 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, None)],
    )
    if not fit.success:
        raise RuntimeError(f"Platt fit did not converge: {fit.message}")
    a, b = fit.x
    return float(a - b * mean / sd), float(b / sd)
