# System library imports.
import numpy as np


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
        p = np.array(p, copy=0)
        if out is None:
            out = np.ndarray(p.shape, dtype=float)
        return np.less(np.random.sample(p.shape), p, out)
