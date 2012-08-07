import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "math.h":
    double exp(double)

def boltzmann_dist(x, int axis=-1, out=None):
    """ Compute the Boltzmann distribution (at fundamental temperature 1) 
    for an array of energies. An extra 'zero-point' energy is included.
    """
    x = np.array(x, copy=0, dtype=np.double)
    if axis < 0:
        axis += x.ndim
    if out is None:
        out = np.empty(x.shape, dtype=np.double)

    cdef int i
    cdef double value, total
    cdef np.flatiter it_x = np.PyArray_IterAllButAxis(x, &axis)
    cdef np.flatiter it_out = np.PyArray_IterAllButAxis(out, &axis)
    cdef int axis_size = x.shape[axis]
    cdef int x_stride = x.strides[axis]
    cdef int out_stride = out.strides[axis]

    while np.PyArray_ITER_NOTDONE(it_x):
        total = 1.0
        for i in range(axis_size):
            val = exp((<double*>(np.PyArray_ITER_DATA(it_x) + i*x_stride))[0])
            (<double*>(np.PyArray_ITER_DATA(it_out) + i*out_stride))[0] = val
            total += val    
            
        for i in range(axis_size):
            (<double*>(np.PyArray_ITER_DATA(it_out) + i*out_stride))[0] /= total

        np.PyArray_ITER_NEXT(it_x)
        np.PyArray_ITER_NEXT(it_out)

    return out
