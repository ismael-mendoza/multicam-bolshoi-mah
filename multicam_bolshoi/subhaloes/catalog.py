import warnings

import numpy as np
from astropy.table import Table
from pminh import minh

from multicam.catalogs import intersect
from multicam.subhaloes import quantities


def create_subhalo_cat(host_ids, minh_file, threshold=1.0 / 1000, log=None):
    """

    Args:
        * minh_file: complete minh catalog and must be read in blocks
        * host_ids corresponds only to host haloes w/ pid == -1
        * threshold: min mass of subhalo considered in `f_sub` calculation relative to
            host halo mass.

    Returns:
        Astropy table with columns 'id', 'mvir', 'f_sub', 'm2' where

        * f_sub = ratio of sum of all subhaloes to mass of host halo.
        * m2 = ratio of most massive subhalo mass to host mass
    """
    assert isinstance(host_ids, np.ndarray)
    assert np.all(np.sort(host_ids) == host_ids)

    # will fill out as we read the blocks.
    fnames = ["id", "mvir", "f_sub", "m2"]
    fdata = [host_ids, *[np.zeros(len(host_ids)) for _ in range(len(fnames) - 1)]]
    host_cat = Table(names=fnames, data=fdata)
    host_cat.sort("id")

    # first we fill out all host masses to avoid confusion later.
    with minh.open(minh_file) as mcat:
        for b in range(mcat.blocks):
            names = ["id", "mvir"]
            ids, mvir = mcat.block(b, names)
            cat = Table(names=names, data=[ids, mvir])
            cat.sort("id")

            # intersect so we only fill out info for host haloes contained in this block.
            # not all host_cat ids are contained in block, not all block ids are in host_cat
            keep = intersect(np.array(cat["id"]), np.array(host_cat["id"]))
            indx_ok = intersect(np.array(host_cat["id"]), np.array(cat[keep]["id"]))

            host_cat["mvir"][indx_ok] = cat[keep]["mvir"]

    unfilled = host_cat["mvir"] == 0
    if sum(unfilled) > 0:
        msg = (
            f"{sum(unfilled)} host IDs out of {len(host_cat)} are not contained in minh catalog "
            f"loaded from file {minh_file.stem}."
        )
        if log:
            print(msg, file=open(log, "a"))
        else:
            warnings.warn(msg)
    host_cat["mvir"][unfilled] = np.nan  # avoid division-by-zero warning below.

    # now calculate subhalo quantities
    with minh.open(minh_file) as mcat:
        for b in range(mcat.blocks):
            names = ["pid", "mvir"]
            pid, mvir = mcat.block(b, names)

            # extract all subhalo masses and parent ids in this block.
            # NOTE: `keep` is NOT used because that will filter out subhalo IDs not in `host_ids`.
            sub_pids = pid[pid != -1]
            sub_mvir = mvir[pid != -1]

            # first calculate total subhalo mass
            # NOTE: If ID in host_cat but not in sub_pids then 0 is returned.
            host_cat["f_sub"] += quantities.m_sub(
                host_cat["id"], host_cat["mvir"], sub_pids, sub_mvir, threshold=threshold
            )

            # and most massive subhalo mass
            m2_curr = quantities.m2_sub(host_cat["id"], sub_pids, sub_mvir).reshape(-1, 1)
            m2_prev = np.array(host_cat["m2"]).reshape(-1, 1)
            m2 = np.hstack([m2_curr, m2_prev]).max(axis=1)
            host_cat["m2"] = m2

    # finally calculate ratios.
    host_cat["f_sub"] = host_cat["f_sub"] / host_cat["mvir"]
    host_cat["m2"] = host_cat["m2"] / host_cat["mvir"]
    return host_cat
