"""Functions to compute gradients from MAH"""
import findiff
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def get_gradient(f, x, k=1, acc=2):
    """
    - f is an array with samples from a function with shape (n_samples, n_features)
    - x is the points where f is evaluated, x is assumed to be a uniform grid (linear spacing)
        Assumed to be the same for all n_samples.
    - k controls the step size.
    - acc is the accuracy at which to take the derivative.
    """

    assert f.shape[1] >= 2 * k
    coeff_table = findiff.coefficients(deriv=1, acc=acc)
    grad = []

    for i in range(f.shape[1]):
        if i - k * (acc // 2) < 0:
            mode = "forward"
            hx = x[i + 1] - x[i]  # accretion rate changes spacing a little bit at some scale.
        elif i - k * (acc // 2) >= 0 and i + k * (acc // 2) < f.shape[1]:
            mode = "center"
            hx = x[i + 1] - x[i]
        else:
            mode = "backward"
            hx = x[i] - x[i - 1]

        coeffs = coeff_table[mode]["coefficients"]  # coefficients are indepenent of step size.
        offsets = coeff_table[mode]["offsets"] * k + i

        assert np.all((offsets < f.shape[1]) & (offsets >= 0))

        deriv = np.sum(f[:, offsets] * coeffs.reshape(1, -1), axis=1) / (hx * k)
        grad.append(deriv.reshape(-1, 1))

    return np.hstack(grad)


def get_savgol_grads(scales, ma, k=5, deriv=1, n_samples=200):
    log_a = np.log(scales)
    log_ma = np.log(ma)
    assert np.sum(np.isnan(log_ma)) == 0
    f_log_ma = interp1d(log_a, log_ma, bounds_error=False, fill_value=np.nan)
    log_a_unif = np.linspace(log_a[0], log_a[-1], n_samples)
    log_ma_unif = f_log_ma(log_a_unif)
    d_log_a = abs(log_a_unif[-1] - log_a_unif[0]) / (len(log_a_unif) - 1)
    gamma_unif = savgol_filter(
        log_ma_unif, polyorder=4, window_length=k, deriv=deriv, delta=d_log_a
    )
    f_gamma = interp1d(log_a_unif, gamma_unif, bounds_error=False, fill_value=np.nan)
    gamma_a = f_gamma(log_a)
    return gamma_a
