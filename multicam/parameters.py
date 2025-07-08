import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_so

cosmology.setCosmology("bolshoi")

default_params = {
    "id",
    "pid",
    "mvir",
    "vvir",
    "rvir",
    "r200m/rvir",
    "r500c/rvir",
    "x",
    "y",
    "z",
    "cvir",
    "x0",
    "v0",
    "spin",
    "q",
    "t/|u|",
    "phi_l",
    "gamma_tdyn",
    "tdyn",
    "vmax/vvir",
    "voff/vvir",
    "scale_of_last_mm",
    "b_to_a",
    "c_to_a",
    "spin_bullock",
}


def derive_vvir(mcat, b):
    # units = km/s
    # prevent overflow by combining MKS factors into one constant.
    # k = G_mks * mvir_factor_mks / rvir_factor_mks
    k = 6.674e-11 * 1.988435e30 / 3.086e19
    rvir, mvir = mcat.block(b, ["rvir", "mvir"])
    vvir_mks = np.sqrt(k * mvir / rvir)
    vvir = vvir_mks / 1e3
    return vvir


def get_vvir(rvir, mvir):
    """Get vvir from rvir and mvir.

    Args:
        rvir: virial radius in kpc/h
        mvir: virial mass in Msun/h
    """
    k = 6.674e-11 * 1.988435e30 / 3.086e19
    vvir_mks = np.sqrt(k * mvir / rvir)
    vvir = vvir_mks / 1e3
    return vvir


def derive(pname: str, mcat, b):
    """Derive additional useful halo properties that are not in .minh catalog."""
    if pname == "phi_l":
        ax, ay, az, jx, jy, jz = mcat.block(b, ["ax", "ay", "az", "jx", "jy", "jz"])
        num = ax * jx + ay * jy + az * jz
        denom = np.sqrt(ax**2 + ay**2 + az**2) * np.sqrt(jx**2 + jy**2 + jz**2)
        return np.arccos(num / denom)

    if pname == "x0":
        xoff, rvir = mcat.block(b, ["xoff", "rvir"])
        return xoff / rvir

    if pname == "v0":
        vvir = derive_vvir(mcat, b)
        [voff] = mcat.block(b, ["voff"])  # km /s
        return voff / vvir

    if pname == "tdyn":
        [rvir] = mcat.block(b, ["rvir"])
        vvir = derive_vvir(mcat, b)
        rvir_mks = rvir * 3.086e19  # rvir in kpc/h not Mpc/h
        vvir_mks = vvir * 1e3
        tdyn_mks = 2 * rvir_mks / vvir_mks
        return tdyn_mks / (365 * 24 * 3600) / 10**9

    if pname == "q":
        b_to_a, c_to_a = mcat.block(b, ["b_to_a", "c_to_a"])
        return (1 / 2) * (b_to_a + c_to_a)

    if pname == "vvir":
        return derive_vvir(mcat, b)

    if pname == "cvir":
        rvir, rs = mcat.block(b, ["rvir", "rs"])
        return rvir / rs

    if pname == "vmax/vvir":
        vvir = derive_vvir(mcat, b)
        [vmax] = mcat.block(b, ["vmax"])
        return vmax / vvir

    if pname == "voff/vvir":
        vvir = derive_vvir(mcat, b)
        [voff] = mcat.block(b, ["voff"])
        return voff / vvir

    if pname == "r200m/rvir":
        [rvir, m200m] = mcat.block(b, ["rvir", "m200b"])
        r200m = mass_so.M_to_R(m200m, 0.0, "200m")
        return r200m / rvir

    if pname == "r500c/rvir":
        [rvir, m500c] = mcat.block(b, ["rvir", "m500c"])
        r500c = mass_so.M_to_R(m500c, 0.0, "500c")
        return r500c / rvir

    else:
        raise NotImplementedError(f"'{pname}' is not avaiable.")
