"""Contains functions to perform and process various fits (e.g. diffmah) to MAH data."""
from pathlib import Path

import numpy as np
from lmfit import Parameters, minimize
from scipy.optimize import curve_fit
from tqdm import tqdm

from multicam.cosmo import get_t_from_a


def get_alpha(zs, lma):
    # use the fit of the form:
    # log m(z) = - \alpha * z
    # get best exponential fit to the line of main progenitors.

    def lma_fit(z, alpha):
        return -alpha * z

    opt_params, _ = curve_fit(lma_fit, zs, lma, p0=(1,))
    return opt_params  # = alpha


def alpha_analysis(ma, scales, mass_bins, alpha_file="../../output/alpha_fits.npy"):
    alpha_file = Path(alpha_file)
    if alpha_file.exists():
        alphas = np.load(alpha_file)
    else:
        # alpha parametrization fit MAH
        # m(a) = exp(- alpha * z)
        alphas = []
        for ii in tqdm(range(len(ma)), desc="Fitting Alpha parametrization"):
            lam_ii = np.log(ma)[ii]
            alpha = get_alpha(1 / scales - 1, lam_ii)
            alphas.append(alpha)
        alphas = np.array(alphas)
        np.save(alpha_file, alphas)
    ma_exp = np.exp(-alphas * (1 / scales - 1))
    am_exp = (1 - (1 / alphas * np.log(mass_bins))) ** -1

    return alphas, ma_exp, am_exp


def softplus(z):
    return np.log(1 + np.exp(z))


def inv_softplus(z):
    return np.log(np.exp(z) - 1)


def get_early_late(ue, ul):
    late = softplus(ul)
    early = late + softplus(ue)
    return early, late


# k = 3.5 is kept constant in Hearin2021
def alpha_diffmah(t, tau_c, alpha_early, alpha_late, k=3.5):
    # t and tau_c in Gyrs
    return alpha_early + (alpha_late - alpha_early) / (1 + np.exp(-k * (t - tau_c)))


def transform_diffmah(x0, beta_e, beta_l):
    tau_c = 10 ** (x0)
    alpha_late = softplus(beta_l)
    alpha_early = alpha_late + softplus(beta_e)
    return tau_c, alpha_early, alpha_late


def fit_hearin_params(ma_peak, scales, sim_name="Bolshoi", do_log=False):
    fit_pars = Parameters()
    fit_pars.add("x0", value=0.1)
    fit_pars.add("beta_e", value=2.08)
    fit_pars.add("beta_l", value=-1.04)

    t = get_t_from_a(scales, sim_name)
    t0 = get_t_from_a(1, sim_name)

    def model(t, t0, x0, beta_l, beta_e):
        # model m(t) = M_peak(t) / M(0) = (t/t0)**(alpha(t))
        # M0 = M_peak(t=0)
        tau_c, alpha_early, alpha_late = transform_diffmah(x0, beta_e, beta_l)
        return (t / t0) ** alpha_diffmah(t, tau_c, alpha_early, alpha_late)

    def residual(pars, data, t, t0):
        vals = pars.valuesdict()
        x0 = vals["x0"]
        beta_l = vals["beta_l"]
        beta_e = vals["beta_e"]

        if do_log:
            return np.log10(data) - np.log10(model(t, t0, x0, beta_l, beta_e))

        else:
            return data - model(t, t0, x0, beta_l, beta_e)

    args = (ma_peak, t, t0)
    try:
        out = minimize(residual, fit_pars, args=args)
        x0 = out.params["x0"].value
        beta_e = out.params["beta_e"].value
        beta_l = out.params["beta_l"].value
        tau_c, alpha_early, alpha_late = transform_diffmah(x0, beta_e, beta_l)
    except ValueError:
        tau_c, alpha_early, alpha_late = np.nan, np.nan, np.nan
    return (tau_c, alpha_early, alpha_late)


def diffmah_analysis(ma_peaks, scales, sim_name="Bolshoi", do_log=False):
    data = []
    for ma_peak in tqdm(ma_peaks, desc="Fitting Diffmah parameters"):
        data.append(fit_hearin_params(ma_peak, scales, sim_name, do_log=do_log))
    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 2]
