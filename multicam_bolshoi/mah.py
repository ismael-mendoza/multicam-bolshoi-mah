"""Functions related to extracting MAH information from output catalogs."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from multicam_bolshoi import catalogs


def get_an_from_am(am, mass_bins, mbin=0.498):
    """Return scale corresponding to first mass bin bigger than `mbin`."""
    idx = np.where(mass_bins > mbin)[0][0]
    return am[:, idx]


def get_mah(
    outdir="../../data/processed/bolshoi_m12",
    cutoff_missing=0.05,
    cutoff_particle=0.05,
    particle_mass=1.35e8,  # Bolshoi
    particle_res=50,
    n_mass_bins=100,
    log_mbin_spacing=True,
):
    """Get catalog, indices, scales from given catalog pipeline outdir path."""
    outdir = Path(outdir)
    cat_file = outdir.joinpath("final_table.csv")
    z_map_file = outdir.joinpath("z_map.json")

    # load all available scales and indices.
    with open(z_map_file, "r", encoding="utf-8") as fp:
        scale_map = json.load(fp)  # map from i -> scale
    indices = np.array(list(scale_map.keys()))
    scales = np.array(list(scale_map.values()))
    # want order: early -> late. Low scales -> high scales (1.0)
    sort_idx = np.argsort(scales)
    indices = indices[sort_idx].astype(int)  # easier to handle.
    scales = scales[sort_idx]

    # load catalog
    cat = catalogs.load_cat_csv(cat_file)

    # extract m(a) information.
    ma_info = _get_ma_info(cat, indices)
    ma = ma_info["ma_vir"]
    ma_peak = ma_info["ma_peak"]

    # get relevant cutoff on scales and mass bins (defined on m(a)).
    min_scale, min_mass_bin = determine_cutoffs(
        ma_info["Mvir"],
        scales,
        cutoff_missing,
        cutoff_particle,
        particle_mass,
        particle_res,
    )

    # fill nan's with average `m` value of 1 particle
    avg_mass = np.nanmean(cat["mvir"])
    ma[np.isnan(ma)] = particle_mass / avg_mass
    ma_peak[np.isnan(ma_peak)] = particle_mass / avg_mass

    # obtain a(m) and corresponding mass_bins
    am, mass_bins = get_am(
        ma, scales, min_mass_bin, n_mass_bins, log_spacing=log_mbin_spacing
    )

    # filter scales, indices, m(a) based on `min_scale`.
    keep_scale = scales > min_scale
    indices = indices[keep_scale]
    scales = scales[keep_scale]
    ma = ma.T[keep_scale].T
    ma_peak = ma_peak.T[keep_scale].T

    # remove haloes with np.nan in a(m)
    keep_am = ~np.isnan(np.sum(am, axis=1))
    cat = cat[keep_am]
    ma = ma[keep_am]
    ma_peak = ma_peak[keep_am]
    am = am[keep_am]

    assert np.sum(np.isnan(ma)) == 0
    assert np.all(ma > 0)
    assert np.sum(np.isnan(am)) == 0

    return {
        "cat": cat,
        "ma": ma,
        "ma_peak": ma_peak,
        "am": am,
        "scales": scales,
        "indices": indices,
        "mass_bins": mass_bins,
        "mpeak": ma_info["Mpeak"][:, -1][keep_am],
    }


def _get_ma_info(cat, indices):
    assert "mvir_a0" in cat.colnames
    assert "mvir_a160" in cat.colnames
    Mvir = np.zeros((len(cat), len(indices)))
    for i, idx in enumerate(indices):
        colname = f"mvir_a{idx}"
        Mvir[:, i] = cat[colname]
    Mpeak = np.fmax.accumulate(
        Mvir, axis=1
    )  # fmax ignores nan's, but keeps the beginning ones.

    ma_vir = Mvir / Mvir[:, -1].reshape(-1, 1)
    ma_peak = Mpeak / Mpeak[:, -1].reshape(-1, 1)
    ma_mix = Mvir / Mpeak[:, -1].reshape(-1, 1)

    return {
        "ma_vir": ma_vir,
        "ma_peak": ma_peak,
        "ma_mix": ma_mix,
        "Mvir": Mvir,
        "Mpeak": Mpeak,
    }


def get_am(ma, scales, min_mass_bin, n_bins=100, log_spacing=True):
    """
    1. Inversion is only a well-defined process for monotonic functions, and m(a) for an
    individual halo isn't necessarily monotonic. To solve this, the standard redefinition of a(m0)
    is that it's the first a where m(a) > m0. (This is, for example, how Rockstar defines halfmass
    scales.)

    2. Next, first pick your favorite set of mass bins that you'll evaluate it at. I think
    logarithmic bins spanning 0.01m(a=1) to 1m(a=1) is pretty reasonable, but you should probably
    choose this based on the mass ranges which are the most informative.

    3. Now, for each halo with masses m(a_i), measure M_peak(a_i) = max_j{ m(a_j) | j <= i}.

    4. Remove (a_i, M_peak(a_i)) pairs where M_peak(a_i) = M_peak(a_{i-1}), since this will mess
    up the inversion.

    5. Use scipy.interpolate.interp1d to create a function, f(m), which evaluates a(m).
    For stability, you'll want to run the interpolation on log(a_i) and log(M(a_i)), not a_i and M
    (a_i).

    6. Evaluate f(m) at the mass bins you decided that you liked in step 2. Now you can run your
    pipeline on this, just like you did for m(a).
    """
    assert np.sum(np.isnan(ma)) == 0, (
        "m(a) needs to be filled with `fill_value` previously."
    )

    # 1. + 2.
    if log_spacing:
        mass_bins = np.linspace(np.log(min_mass_bin), np.log(1.0), n_bins)
    else:
        mass_bins = np.log(np.linspace(min_mass_bin, 1.0, n_bins))

    # 3.
    # NOTE: Make function monotonic. We assume start is early -> late ordering.
    # NOTE: ma should not have any cuts.

    # fmax ignores nan's (except beginning ones)
    m_peak = np.fmax.accumulate(ma, axis=1)

    # 4. + 5.
    fs = []
    for i in range(len(m_peak)):  # pylint: disable=consider-using-enumerate
        pairs = [(scales[0], m_peak[i][0])]
        count = 0
        for j in range(1, len(m_peak[i])):
            # keep only pairs that do NOT satisfy (a_{j-1}, Ma_{j-1}) = (a_j, Ma_j)
            if pairs[count][1] != m_peak[i][j]:
                pairs.append((scales[j], m_peak[i][j]))
                count += 1

        assert len(pairs) != 1, (
            "Only 1 pair added, so max reached at a -> 0, impossible."
        )

        _scales = np.array([pair[0] for pair in pairs])
        _m_peaks = np.array([pair[1] for pair in pairs])
        fs.append(
            interp1d(
                np.log(_m_peaks), np.log(_scales), bounds_error=False, fill_value=np.nan
            )
        )

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)


def determine_cutoffs(
    Mvir,
    scales,
    cutoff_missing=0.05,
    cutoff_particle=0.05,
    particle_mass=1.35e8,
    particle_res=50,
):
    """Return minimum scale to use for m(a) and minimum mass bin to use for a(m).

    Args:
        * Mvir: Raw virial mass accretion history, array w/ shape (n_samples, n_scales).

        * cutoff_particle: is the percentage of haloes at a given scale that we tolerate
            with < `particle_res` particles,

        * cutoff_missing: the percentage of haloes with no progenitors we are OK discarding.
            NOTE: Phil suggested to make this small so that we don't bias our samples
            (late-forming haloes are more likely to have nan's at the end.)

    """
    # minimum virial mass we are comfortable with
    min_mass = particle_res * particle_mass

    # NOTE: Assumes that order is early -> late.
    # determine scales we should cutoff based on percentage of NaN progenitors.
    n_nan = np.sum(np.isnan(Mvir), axis=0) / len(Mvir)
    idx1 = np.where(n_nan > cutoff_missing)[0][-1] + 1
    min_scale1 = scales[idx1]

    # NOTE: Assumes that order is early -> late.
    # NOTE: These two cuts are different because ROCKSTAR does not limit to ~50 particles.
    # determine scale we should cutoff based on num. of particles.
    # it is monotonically decreasing
    m_cutoff = np.nanquantile(Mvir, cutoff_particle, axis=0)
    idx2 = np.where(m_cutoff < min_mass)[0][-1] + 1  # over scales NOT data points.
    min_scale2 = scales[idx2]

    # combine two criteria into one scale
    min_scale = max(min_scale1, min_scale2)

    # Now we want to do the same but for mass bins.
    # NOTE: The explanation is that we want (1) minimum mass bin m0 such that at least
    # 99% haloes have a progenitor at that mass bin and (2) minimum mass bin m0 such that
    # 90% of haloes have their earliest a0 s.t. m(a0) > m0 satisfy m(a0) * Mvir(a=1)/ 1.35e8 > 50
    # NOTE: (1) is exactly Phil's suggestin.
    min_mass_bin1 = np.nanquantile(
        np.nanmin(Mvir, axis=1) / Mvir[:, -1], 1 - cutoff_missing
    )
    min_mass_bin2 = np.nanquantile(min_mass / Mvir[:, -1], 1 - cutoff_particle)
    assert isinstance(min_mass_bin1, float)
    assert isinstance(min_mass_bin2, float)
    min_mass_bin = max(min_mass_bin1, min_mass_bin2)

    return min_scale, min_mass_bin
