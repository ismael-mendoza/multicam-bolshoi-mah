"""Functions used to calculate correlations and errors on correlations."""
import numpy as np
from scipy import stats


def spearmanr(*args, **kwargs):
    return stats.spearmanr(*args, **kwargs).correlation


def get_2d_corr(x, y, ibox):
    assert len(x.shape) == 2 and len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    m = x.shape[1]
    corrs = np.zeros(m)
    errs = np.zeros(m)
    for jj in range(m):
        x_j = x[:, jj]
        corrs[jj] = spearmanr(x_j, y)
        errs[jj] = vol_jacknife_err(spearmanr, ibox, x_j, y)

    return corrs, errs


def get_opt_corr(ma, y, bins, ibox):
    def _get_opt_indx(ma, y):
        m = ma.shape[1]
        corrs = np.zeros(m)
        for jj in range(m):
            corrs[jj] = spearmanr(ma[:, jj], y)
        max_indx = np.nanargmax(abs(corrs))
        return max_indx

    def _get_opt_scale(ma, y, bins=None):
        assert bins is not None
        max_indx = _get_opt_indx(ma, y)
        return bins[max_indx]

    max_indx = _get_opt_indx(ma, y)
    err_opt_bin = vol_jacknife_err(_get_opt_scale, ibox, ma, y, bins=bins)
    return max_indx, bins[max_indx], err_opt_bin


def add_box_indices(cat, boxes=8, box_size=250):
    # box_size is in Mpc
    # create a new row add it to the catalogue for which box it is in.
    assert int(boxes ** (1.0 / 3)) == boxes ** (1.0 / 3)
    box_per_dim = int(boxes ** (1.0 / 3))
    divides = np.linspace(0, box_size, box_per_dim + 1)[1:-1]  # only use the middle.
    cat.add_column(np.zeros(len(cat)), name="ibox")
    for k, dim in enumerate(["x", "y", "z"]):
        for d in divides:
            cat["ibox"] += 2**k * (d < cat[dim])


def vol_jacknife_err(fn, ibox, *args, **kwargs):
    n_boxes = int(np.max(ibox) + 1)
    values = []
    for b in range(n_boxes):
        box_keep = ibox != b
        bargs = (x[box_keep] for x in args)
        value = fn(*bargs, **kwargs)
        values.append(value)
    values = np.array(values)
    return np.sqrt(values.var(axis=0) * (n_boxes - 1))
