cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_ints(long[:] x, long[:] out_bins, long[:] out_bin_edges):
    cdef long n_bins = len(out_bin_edges) - 1
    cdef long n = len(x)
    cdef long i

    assert(len(x) == len(out_bins))

    for i in range(n_bins + 1):
        out_bin_edges[i] = 0

    cdef long bin
    for i in range(n):
        bin = x[i]
        if bin < 0 or bin >= n_bins - 1: continue
        out_bin_edges[bin + 2] += 1
    
    for i in range(2, n_bins + 1):
        out_bin_edges[i] += out_bin_edges[i - 1]

    cdef long idx
    for i in range(n):
        bin = x[i]
        if bin < 0 or bin >= n_bins: continue
        idx = out_bin_edges[bin+1]
        out_bin_edges[bin+1] += 1
        out_bins[idx] = i

    return out_bins, out_bin_edges

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_2d(double[:,:] points, long n_bins, double L,
           long[:] idx, long[:] out_bins, long[:] out_bin_edges):
    
    cdef double[:] x = points[0]
    cdef double[:] y = points[1]
    cdef long n = len(x)

    assert(n == len(out_bins))
    assert(n == len(idx))
    assert(n_bins*n_bins + 1 == len(out_bin_edges))

    cdef double dx = L / n_bins
    cdef long ix, iy

    for i in range(n):
        ix = <long>(x[i] / dx)
        iy = <long>(y[i] / dx)
        if ix < 0 or ix >= n_bins or iy < 0 or iy >= n_bins: continue

        idx[i] = ix + iy*n_bins

    return bin_ints(idx, out_bins, out_bin_edges)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bin_3d(double[:,:] points, long n_bins, double L,
           long[:] idx, long[:] out_bins, long[:] out_bin_edges):
    
    cdef double[:] x = points[0]
    cdef double[:] y = points[1]
    cdef double[:] z = points[2]
    cdef long n = len(x)

    assert(n == len(out_bins))
    assert(n == len(idx))
    assert(n_bins*n_bins*n_bins + 1 == len(out_bin_edges))

    cdef double dx = L / n_bins
    cdef long ix, iy, iz

    for i in range(n):
        ix = <long>(x[i] / dx)
        iy = <long>(y[i] / dx)
        iz = <long>(z[i] / dx)
        if ((ix < 0 or ix >= n_bins) or
            (iy < 0 or iy >= n_bins) or
            (iz < 0 or iz >= n_bins)): continue

        idx[i] = ix + iy*n_bins + iz*n_bins*n_bins

    return bin_ints(idx, out_bins, out_bin_edges)
