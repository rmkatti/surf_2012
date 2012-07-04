import numpy as np

def bit_vector_from_str(s):
    """ Convert a string to a bit vector.
    """
    if np.isscalar(s):
        return np.array(map(int, s))
    s = np.array(s, copy=0)
    length = len(s.flat[0])
    v = s.view(dtype='|S1').reshape(s.shape + (length,))
    return v.astype(int)

def bit_vector_to_str(v):
    """ Convert a bit vector to a string.
    """
    v = np.array(v, copy=0)
    s = np.apply_along_axis(''.join, -1, v.astype(str))
    return str(s) if s.shape == () else s

def count_bit_vectors(v):
    """ Given an array of bit vectors, count the occurences of each vector.

    Returns the pair (array of unique vectors, array of corresponding counts).
    """
    v = v[np.lexsort(v.T[::-1])] # Sort vectors lexicographically.
    diff = np.ones(v.shape, v.dtype)
    diff[1:] = np.diff(v, axis=0)
    idx = np.where(np.any(diff, axis=1))[0]
    unique_v = v[idx]
    counts = np.diff(np.append(idx, v.shape[0]))
    return unique_v, counts

try:
    from _util import logistic, sigmoid, sample_indicator
except ImportError:
    print 'WARNING: Optimized math functions not available.'

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
