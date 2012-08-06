# System library imports.
import numpy as np

# Optimized ufuncs.
try:
    from _math import logit, logistic, sigmoid, \
        sample_exclusive_indicators, sample_indicator

except ImportError:
    print 'WARNING: Optimized math functions not available.'

    def logit(p):
        """ The standard logit (log-odds) function. """
        return np.log(p / (1-p))

    def logistic(x):
        """ The standard logistic (sigmoid) function. """
        return 1 / (1 + np.exp(-x))
    sigmoid = logistic

    def sample_exclusive_indicators(p, axis=-1, out=None):
        """ Samples a bit vector of mutually exclusive indicator variables.
        """
        def sample_exclusive_indicators_1d(p):
            cdf = p.cumsum()
            idx = cdf.searchsorted(np.random.random(), side='right')
            out = np.zeros(p.size)
            if idx < out.size:
                out[idx] = 1.0
            return out

        # XXX: apply_along_axis does not support 'out'.
        s = np.apply_along_axis(sample_exclusive_indicators_1d, axis, p)
        if out is None: out = s
        else: out[:] = s
        return out

    def sample_indicator(p, out=None):
        """ Yields 1 with probability p and 0 with probability 1-p. 
        """
        p = np.array(p, dtype=np.double, copy=0)
        if out is None:
            out = np.empty(p.shape, dtype=np.double)
        return np.less(np.random.random(p.shape), p, out)
