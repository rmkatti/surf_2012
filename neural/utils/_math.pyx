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

cdef double sample_indicator_d(double p):
    return 1.0 if gsl_rng_uniform(rng) < p else 0.0

sample_indicator = register_ufunc_d(sample_indicator_d, 'sample_indicator', '''\
Yields 1.0 with probability x and 0.0 with probability 1-x.
''')