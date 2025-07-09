from __future__ import division, print_function

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


def m_sub(host_ids, host_mvir, sub_pids, sub_mvir, threshold=1.0 / 1000):
    """M_sub returns the sum of the mass of all subhaloes of each host."""
    bins = bin_by_host(host_ids, sub_pids)
    M_sub = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_mvir_i = sub_mvir[bins[i]]
        keep = sub_mvir_i > threshold * host_mvir[i]
        M_sub[i] = np.sum(sub_mvir_i[keep])

    return M_sub


def m2_sub(host_ids, sub_pids, sub_mvir):
    """Returns mass of most massive subhalo."""
    bins = bin_by_host(host_ids, sub_pids)
    m2 = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_mvir_i = sub_mvir[bins[i]].reshape(-1)
        m2[i] = sub_mvir_i.max() if len(sub_mvir_i) > 0 else 0.0

    return m2


def n_sub(host_ids, sub_pids):
    """How many subhaloes does each halo have?"""
    bins = bin_by_host(host_ids, sub_pids)
    N_sub = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_index = bins[i]
        N_sub[i] = len(sub_index)
    return N_sub


# FIXME: max fails if empty.
def mass_gap(host_mvir, host_ids, sub_mvir, sub_pids):
    bins = bin_by_host(host_ids, sub_pids)
    M_sub = m_sub(host_ids, sub_pids, sub_mvir)
    mgap = np.zeros(len(bins))
    for i in range(len(M_sub)):
        sub_index = bins[i]
        mgap[i] = host_mvir[i] - np.max(sub_mvir[sub_index])
    return mgap
