import numpy as np

def kl_divergence(p, q, epsilon=1e-6):
    """ Computes the Kullback-Leibler divergence D(p||q) between the
    distributions p and q. 

    Parameters:
    -----------
    p, q : array-like
        Arrays of corresponding probabilities; must be of equal dimension.

    epsilon : float, optional
        Probabilities less than 'epsilon' are replaced by 'epsilon' to prevent
        the KL divergence from blowing up.
    """
    p = np.max(p, epsilon)
    q = np.max(q, epsilon)
    return np.sum(p * np.log(p/q))

def sample_indicator(p):
    """ Yields 1 with probability p and 0 with probability 1-p. """
    p = np.array(p, copy=0)
    return np.array(np.random.sample(p.shape) < p, 
                    dtype=np.float)
