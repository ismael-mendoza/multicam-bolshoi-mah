import numpy as np
from diffmah.fit_mah_helpers import get_loss_data, log_mah_mse_loss_and_grads
from diffmah.utils import jax_adam_wrapper
from tqdm import tqdm

from multicam.cosmo import get_t_from_a
from multicam.mah import get_mah

mah_data = get_mah("m12", "output", cutoff_missing=0.05, cutoff_particle=0.05)

ma_peak = mah_data["ma_peak"]
mpeak = mah_data["mpeak"]
scales = mah_data["scales"]
t = get_t_from_a(scales)
t0 = get_t_from_a(1)

log_mah = np.log10(ma_peak * mpeak.reshape(-1, 1))
log_min_mass = np.log10(100 * 1.35e8)


fit_params = []

for ii in tqdm(range(log_mah.shape[0])):
    n_steps = 200
    lma_i = log_mah[ii, :]
    p_init, loss_data = get_loss_data(t, lma_i, log_min_mass)
    _res = jax_adam_wrapper(log_mah_mse_loss_and_grads, p_init, loss_data, n_steps, n_warmup=1)
    p_best, loss, loss_arr, params_arr, fit_terminates = _res
    fit_params.append(tuple(p_best))

    if ii % 100 == 0 and ii != 0:
        np.save(f"output/pbest_{ii}", np.array(fit_params))
