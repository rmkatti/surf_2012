# System library imports.
import numpy as np

# Optimized ufuncs.
try:
    from _math import logit, logistic, sigmoid, sample_indicator

except ImportError:
    print 'WARNING: Optimized math functions not available.'

    def logit(p):
        """ The standard logit (log-odds) function. """
        return np.log(p / (1-p))

    def logistic(x):
        """ The standard logistic (sigmoid) function. """
        return 1 / (1 + np.exp(-x))
    sigmoid = logistic

    def sample_indicator(p, out=None):
        """ Yields 1 with probability p and 0 with probability 1-p. """
        p = np.array(p, dtype=np.double, copy=0)
        if out is None:
            out = np.empty(p.shape, dtype=np.double)
        return np.less(np.random.random(p.shape), p, out)


def sample_discrete(p, size=None):
    """ Samples from an arbitrary discrete distribution.
    """
    p = np.array(p, dtype=np.double, ndmin=1, copy=0)
    if p.ndim != 1:
        raise ValueError("p must be 1-dimensional")

    cdf = p.cumsum()
    idx = cdf.searchsorted(np.random.random(size), side='right')
    return idx
