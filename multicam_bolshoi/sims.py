"""Collection of all simulations that could be used with this package."""
from collections import namedtuple

Sim = namedtuple(
    "Simulation",
    [
        "name",
        "box_size",  # Mpc/h
        "n_particles",  # counts
        "particle_mass",  # Msun/h
        "force_resolution",  # kpc/h
        "initial_redshift",  # unitless
        "h",  # unitless
        "omega_lambda",  # unitless
        "omega_m",  # unitlesss
        "omega_b",  # unitless
        "sigma8",  # unitless
        "n",  # normalization of the Power spectrum
    ],
)

# https://www.cosmosim.org/cms/simulations/bolshoi/
Bolshoi = Sim(
    name="Bolshoi",
    box_size=250,
    n_particles=2048**3,
    particle_mass=1.35e8,
    force_resolution=1.0,
    initial_redshift=80,
    h=0.70,
    omega_lambda=0.73,
    omega_m=0.27,
    omega_b=0.0469,
    sigma8=0.82,
    n=0.95,
)

# https://www.cosmosim.org/cms/simulations/bolshoip/
BolshoiP = Sim(
    name="BolshoiP",
    box_size=250,
    n_particles=2048**3,
    particle_mass=1.55e8,
    force_resolution=1.0,
    initial_redshift=80,
    h=0.70,
    omega_lambda=0.69289,
    omega_m=0.30711,
    omega_b=0.048,
    sigma8=0.96,
    n=0.82,
)

# https://www.cosmosim.org/cms/simulations/mdpl2/
MDPL2 = Sim(
    name="MDPL2",
    box_size=1e3,
    n_particles=3840**3,
    particle_mass=1.51e9,
    force_resolution=(5, 13),  # low redshift and high redshift respectively
    initial_redshift=120,
    h=0.6777,
    omega_lambda=0.692885,
    omega_m=0.307115,
    omega_b=0.048206,
    sigma8=0.96,
    n=0.8228,
)

all_sims = {sim.name: sim for sim in [Bolshoi, BolshoiP, MDPL2]}
