include "external/numpy_ufuncs.pxi"

cdef extern from "math.h":
    double exp(double)
    float expf(float)

cdef float logistic_f(float x):
    return 1.0 / (1.0 + expf(-x))

cdef double logistic_d(double x):
    return 1.0 / (1.0 + exp(-x))

doc = '''\
The standard logistic (sigmoid) function, the inverse of the logit (log odds)
function.
'''
logistic = register_ufunc_fd(logistic_f, logistic_d, "logistic", doc)
sigmoid = logistic
