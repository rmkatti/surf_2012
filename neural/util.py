import numpy as np

def bit_vector(n, l):
    """ Convert a nonnegative integer 'n' to a binary array of length 'l'.
    """
    if not np.isscalar(n):
        n = np.array(n, copy=0)[:,np.newaxis]
    # Note: For a general radix r, this is
    #   return (n / r**np.arange(l-1,-1,-1)) % r.
    return 1 & (n / 2**np.arange(l-1,-1,-1))

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
    
def sigmoid(x):
    """ The standard sigmoid function. """
    return 1 / (1 + np.exp(-x))
