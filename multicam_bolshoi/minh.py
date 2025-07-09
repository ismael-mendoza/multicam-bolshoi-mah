"""Functions that use minh"""
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from pminh import minh

from multicam import parameters
from multicam.catalogs import filter_cat


def load_cat_minh(minh_file: str, params: list, filters: dict, verbose=False):
    """Return astropy table of Halo present-day parameters from .minh catalog.

    Parameters are filtered on the fly to avoid memory errors.
    """
    assert Path(minh_file).name.endswith(".minh")
    assert set(filters.keys()).issubset(set(params))
    if verbose:
        warnings.warn("Divide by zero errors are ignored, and filtered out.")

    mcat = minh.open(minh_file)
    cats = []
    for b in range(mcat.blocks):
        cat = Table()

        # obtain all params from minh and their values.
        with np.errstate(divide="ignore", invalid="ignore"):
            for param in params:
                if param in mcat.names:
                    [value] = mcat.block(b, [param])
                else:
                    value = parameters.derive(param, mcat, b)
                cat.add_column(value, name=param)

        # make sure it's sorted by ID in case using id_filter
        cat.sort("id")

        # filter to reduce size of each block.
        cat = filter_cat(cat, filters)
        cats.append(cat)

    fcat = vstack(cats)
    fcat.sort("id")

    return fcat
