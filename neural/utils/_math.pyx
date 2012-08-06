import numpy as np
cimport numpy as np

np.import_array()
include "../external/numpy_ufuncs.pxi"

cdef extern from "math.h":
    double exp(double)
    float expf(float)
    double log(double)
    float logf(float)

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    double gsl_rng_uniform(gsl_rng * r)

cdef gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937)

cdef float logit_f(float p):
    return logf(p / (1.0 - p))

cdef double logit_d(double p):
    return log(p / (1.0 - p))

logit = register_ufunc_fd(logit_f, logit_d, 'logit', '''\
The standard logit (log-odds) function, the inverse of the logistic (sigmoid)
function.
''')

cdef float logistic_f(float x):
    return 1.0 / (1.0 + expf(-x))

cdef double logistic_d(double x):
    return 1.0 / (1.0 + exp(-x))

logistic = register_ufunc_fd(logistic_f, logistic_d, 'logistic', '''\
The standard logistic (sigmoid) function, the inverse of the logit (log odds)
function.
''')
sigmoid = logistic

def sample_exclusive_indicators(p, int axis=-1, out=None):
    """ Samples a bit vector of mutually exclusive indicator variables.
    """
    p = np.array(p, copy=0, dtype=np.double)
    if axis < 0:
        axis += p.ndim
    if out is None:
        out = np.empty(p.shape, dtype=np.double)

    cdef int i, sample_idx
    cdef double cdf, rand
    cdef np.flatiter it_p = np.PyArray_IterAllButAxis(p, &axis)
    cdef np.flatiter it_out = np.PyArray_IterAllButAxis(out, &axis)
    cdef int axis_size = p.shape[axis]
    cdef int p_stride = p.strides[axis]
    cdef int out_stride = out.strides[axis]

    while np.PyArray_ITER_NOTDONE(it_p):
        cdf = 0.0
        rand = gsl_rng_uniform(rng)
        sample_idx = axis_size
        for i in range(axis_size):
            cdf += (<double*>(np.PyArray_ITER_DATA(it_p) + i * p_stride))[0]
            if rand < cdf:
                sample_idx = i
                break
                
        for i in range(axis_size):
            (<double*>(np.PyArray_ITER_DATA(it_out) + i * out_stride))[0] = \
                1.0 if i == sample_idx else 0.0

        np.PyArray_ITER_NEXT(it_p)
        np.PyArray_ITER_NEXT(it_out)

    return out

cdef double sample_indicator_d(double p):
    return 1.0 if gsl_rng_uniform(rng) < p else 0.0

sample_indicator = register_ufunc_d(sample_indicator_d, 'sample_indicator', '''\
Yields 1.0 with probability x and 0.0 with probability 1-x.
''')
