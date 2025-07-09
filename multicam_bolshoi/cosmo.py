"""Functions used to compute cosmological values."""
from astropy.cosmology import LambdaCDM, z_at_value

from multicam.sims import all_sims


def get_fractional_tdyn(scale, tdyn, sim_name="Bolshoi"):
    """Converts scale to fractional Tdyn"""
    sim = all_sims[sim_name]

    # get cosmology based on given sim
    cosmo = LambdaCDM(H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m)

    # tdyn in Gyrs
    z = (1 / scale) - 1
    return (cosmo.age(0).value - cosmo.age(z).value) / tdyn


def get_a_from_t(t: float, sim_name="Bolshoi", method="Bounded"):
    sim = all_sims[sim_name]

    # get cosmology based on given sim
    cosmo = LambdaCDM(H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m)
    z = z_at_value(cosmo.age, t, method=method)  # t in Gyrs
    return 1 / (1 + z)


def get_t_from_a(scale, sim_name="Bolshoi"):
    sim = all_sims[sim_name]

    # get cosmology based on given sim
    cosmo = LambdaCDM(H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m)
    z = (1 / scale) - 1
    return cosmo.age(z).value
