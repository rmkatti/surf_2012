import numpy as np

def choice(a, size=1, p=None):
    """ Generates a random sample from a given 1-D array

    Simplified version of np.random.choice from (as yet unreleased) NumPy 1.7.
    """
    # Format and verify input.
    if isinstance(a, int):
        if a > 0:
            pop_size = a # population size
        else:
            raise ValueError("a must be greater than 0")
    else:
        a = np.array(a, ndmin=1, copy=0)
        if a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        pop_size = a.size
        if pop_size is 0:
            raise ValueError("a must be non-empty")

    if None != p:
        p = np.array(p, dtype=np.double, ndmin=1, copy=0)
        if p.ndim != 1:
            raise ValueError("p must be 1-dimensional")
        if p.size != pop_size:
            raise ValueError("a and p must have same size")
        if any(p < 0):
            raise ValueError("probabilities are not non-negative")
        if not np.allclose(p.sum(), 1):
            raise ValueError("probabilities do not sum to 1")

    # Actual sampling.
    if None != p:
        cdf = p.cumsum()
        cdf /= cdf[-1]
        uniform_samples = np.random.random(size)
        idx = cdf.searchsorted(uniform_samples, side='right')
    else:
        idx = np.random.randint(0, pop_size, size=size)

    # Use samples as indices for a if a is array-like.
    if isinstance(a, int):
        return idx
    else:
        return a.take(idx)

def sample_indicator(p):
    """ Yields 1 with probability p and 0 with probability 1-p. """
    p = np.array(p, copy=0)
    return np.array(np.random.sample(p.shape) < p, 
                    dtype=np.float)
