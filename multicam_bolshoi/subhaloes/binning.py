from __future__ import print_function

import pyximport

pyximport.install()
import time

import numpy as np

from multicam.subhaloes import cy_binning

""" binning.py contains wrappers for Cython routines which bin data almost a
hundred times faster than any native Python routine can.

@author: phil-mansfield
"""


def bin_ints(idx, n_bins, out_bins=None, out_bin_edges=None):
    """bin_ints collects an array of bin indices, idx, into a set of
    n_bins bins. bin_ints returns a list of numpy arrays which each contain the
    indices of the elements in idx which are contained in that bin. If n_bins
    isn't supplied max(idx) + 1 will be used as n_bins.

    If you have memory constraints, you can supply two integer arrays to prevent
    bin_ints from making any allocations. If so, out_bins must be the same
    length as idx and out_bin_edges must be length n_bins + 1.

    All arrays must have dtype=int.
    """

    assert idx.dtype == int

    # Set up out_bins
    if out_bins is None:
        out_bins = np.zeros(len(idx), dtype=int)
    else:
        assert len(out_bins) >= len(idx)
        assert out_bins.dtype == int
        out_bins = idx[: len(idx)]

    # Set up out_bin_edges
    if out_bin_edges is None:
        out_bin_edges = np.zeros(n_bins + 1, dtype=int)
    else:
        assert n_bins + 1 <= len(out_bin_edges)
        assert out_bin_edges.dtype == int
        out_bin_edges = out_bin_edges[: n_bins + 1]

    # Run the Cython routine
    cy_binning.bin_ints(idx, out_bins, out_bin_edges)
    out = [None] * n_bins

    # Convert to a list
    for i in range(n_bins):
        out[i] = out_bins[out_bin_edges[i] : out_bin_edges[i + 1]]

    return out


class Bin2DWorkspace(object):
    def __init__(self, n, n_bins):
        self.n, self.n_bins = n, n_bins
        self.out_idx = np.zeros(n, dtype=int)
        self.out_bins = np.zeros(n, dtype=int)
        self.out_bin_edges = np.zeros(n_bins * n_bins + 1, dtype=int)


def bin_2d(points, n_bins, L, workspace=None):
    assert points.dtype == float
    assert len(points.shape) == 2
    assert points.shape[0] == 2

    n = points.shape[1]

    # Set up workspace.
    if workspace is None:
        workspace = Bin2DWorkspace(n, n_bins)
    assert n == workspace.n
    assert n_bins == workspace.n_bins

    # Run Cython routine.
    n_bins = int(n_bins)
    L = float(L)
    cy_binning.bin_2d(
        points,
        n_bins,
        L,
        workspace.out_idx,
        workspace.out_bins,
        workspace.out_bin_edges,
    )
    out = [None] * (n_bins * n_bins)

    # Convert to a list.
    for i in range(n_bins * n_bins):
        out[i] = workspace.out_bins[workspace.out_bin_edges[i] : workspace.out_bin_edges[i + 1]]

    return out


class Bin3DWorkspace(object):
    def __init__(self, n, n_bins):
        self.n, self.n_bins = n, n_bins
        self.out_idx = np.zeros(n, dtype=int)
        self.out_bins = np.zeros(n, dtype=int)
        self.out_bin_edges = np.zeros(n_bins * n_bins * n_bins + 1, dtype=int)


def bin_3d(points, n_bins, L, workspace=None):
    assert points.dtype == float
    assert len(points.shape) == 2
    assert points.shape[0] == 3

    n = points.shape[1]

    # Set up workspace.
    if workspace is None:
        workspace = Bin3DWorkspace(n, n_bins)
    assert n == workspace.n
    assert n_bins == workspace.n_bins

    # Run Cython routine.
    n_bins = int(n_bins)
    L = float(L)
    cy_binning.bin_3d(
        points,
        n_bins,
        L,
        workspace.out_idx,
        workspace.out_bins,
        workspace.out_bin_edges,
    )
    out = [None] * (n_bins * n_bins * n_bins)

    # Convert to a list.
    for i in range(n_bins * n_bins * n_bins):
        out[i] = workspace.out_bins[workspace.out_bin_edges[i] : workspace.out_bin_edges[i + 1]]

    return out


def main():
    use_workspace = True

    L = 250.0
    n = 1000000

    points = np.random.rand(3, n) * L

    for n_bins in [2, 5, 10, 20, 50, 100]:
        if use_workspace:
            workspace = Bin3DWorkspace(n, n_bins)
        else:
            workspace = None

        t0 = time.time()
        bins = bin_3d(points, n_bins, L, workspace=workspace)
        t1 = time.time()
        bin_sizes = np.array([len(bin) for bin in bins])

        print("Bins = %d^3" % n_bins)
        exp_mean = float(n) / n_bins**3
        exp_std = np.sqrt(exp_mean)
        print("Expected mean = %.4g, expected std = %.3g" % (exp_mean, exp_std))
        print("Actual mean =   %.4g, actual std =   %.3g" % (np.mean(bin_sizes), np.std(bin_sizes)))
        print("dt = %.3g" % (t1 - t0))
        print()


if __name__ == "__main__":
    main()
