from __future__ import division
from __future__ import print_function

import numpy as np

from . import binning

"""
subhalo.py provides several utility functions for analysing a host halo's
subhaloes.

@author: Phil Mansfield
"""


class LookupTable(object):
    """LookupTable creates a fast lookup table which can quickly associated
    subhaloes pids with host halo ids.
    """

    def __init__(self, host_ids):
        self.host_ids = host_ids
        self.host_index = np.arange(len(host_ids))

        order = np.argsort(host_ids)
        self.host_ids = self.host_ids[order]
        self.host_index = self.host_index[order]

    def lookup(self, sub_pids):
        """lookup returns the indices of the host halos of each subhalo.
        Returns -1 if a PID corresponds to a halo not in host_ids.
        """

        # where do we need to insert each id in sub_pids to keep host_ids ordered.
        sub_index = np.searchsorted(self.host_ids, sub_pids)

        # only ones that are smaller than biggest host_id
        sub_ok = sub_index < len(self.host_ids)

        # first select, all indices we know so far are ok in host_ids. Note that the ordering of
        # sub_index is the same ordering as sub_pids.
        # for each 'ok' index indx in sub_index, the ordering is such that id host_index[indx]
        # should match the corresponding 'ok' id in sub_pids if these ids are to be equal.
        sub_ok[sub_ok] &= self.host_ids[sub_index[sub_ok]] == sub_pids[sub_ok]

        # for each sub_index, return the id of the host halo if that id was 'ok', -1 otherwise.
        out = np.ones(len(sub_index), dtype=int) * -1
        out[sub_ok] = self.host_index[sub_index[sub_ok]]

        return out


def bin_by_host(host_ids, sub_pids):
    """bin_by_host bins subhaloes according to their hosts. This function
    returns a list where each element is an array of indices into `sub_pids`
    the correspond to that that host's subhaloes.
    """

    table = LookupTable(host_ids)
    sub_index = table.lookup(sub_pids)
    return binning.bin_ints(sub_index, len(host_ids))


# Provided as an example of how to use bin_by_host:


def m_sub(host_ids, sub_pids, sub_mvir):
    """M_sub returns the sum of the mass of all subhaloes of each host."""
    bins = bin_by_host(host_ids, sub_pids)

    M_sub = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_mvir_i = sub_mvir[bins[i]]
        M_sub[i] = np.sum(sub_mvir_i)

    return M_sub


def n_sub(host_ids, sub_pids):
    """How many subhaloes does each halo have?"""
    bins = bin_by_host(host_ids, sub_pids)
    N_sub = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_index = bins[i]
        N_sub[i] = len(sub_index)
    return N_sub


def m2_sub(host_ids, sub_mvir, sub_pids):
    bins = bin_by_host(host_ids, sub_pids)
    M_sub = m_sub(host_ids, sub_pids, sub_mvir)
    m2 = np.zeros(len(bins))
    for i in range(len(M_sub)):
        sub_index = bins[i]
        m2[i] = np.max(sub_mvir[sub_index])
    return m2


def mass_gap(host_mvir, host_ids, sub_mvir, sub_pids):
    bins = bin_by_host(host_ids, sub_pids)
    M_sub = m_sub(host_ids, sub_pids, sub_mvir)
    mgap = np.zeros(len(bins))
    for i in range(len(M_sub)):
        sub_index = bins[i]
        mgap[i] = host_mvir[i] - np.max(sub_mvir[sub_index])
    return mgap


def run_tests():
    host_ids = np.array([100, 200, 50, -17], dtype=int)
    sub_pids = np.array([100, -17, 100, 200, -17, 100, 75, 300, -20], dtype=int)
    sub_mvir = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

    # test LookupTable

    table = LookupTable(host_ids)
    sub_index = table.lookup(sub_pids)

    assert np.all(sub_index == np.array([0, 3, 0, 1, 3, 0, -1, -1, -1]))

    # test bin_by_host

    bins = bin_by_host(host_ids, sub_pids)

    assert len(bins) == 4
    assert np.all(bins[0] == np.array([0, 2, 5]))
    assert np.all(bins[1] == np.array([3]))
    assert len(bins[2]) == 0
    assert np.all(bins[3] == np.array([1, 4]))

    # test M_sub

    M_sub = m_sub(host_ids, sub_mvir, sub_pids)

    assert np.all(M_sub == np.array([10, 4, 0, 7]))

    print("Test passes!")


if __name__ == "__main__":
    run_tests()
