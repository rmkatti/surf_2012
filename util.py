import numpy as np

def bit_vector(n, l):
    """ Convert a nonnegative integer 'n' to a binary array of length 'l'.
    """
    if not np.isscalar(n):
        n = np.array(n, copy=0)[:,np.newaxis]
    # Note: For a general radix r, this is
    #   return (n / r**np.arange(l-1,-1,-1)) % r.
    return 1 & (n / 2**np.arange(l-1,-1,-1))

def encode_bit_vector(x):
    """ Encode a bit vector as a nonnegative integer.
    """
    return np.sum(x * 2**np.arange(x.shape[-1]-1, -1, -1), axis=-1)

def sigmoid(x):
    """ The standard sigmoid function. """
    return 1 / (1 + np.exp(-x))
