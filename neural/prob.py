""" Generic tools for probabilistic calculations, particularly over binary
valued distributions.
"""
# Standard library imports.
from itertools import chain, izip

# System library imports.
import numpy as np
from scipy.stats import rv_discrete

# Local imports.
from util import bit_vector_to_str, bit_vector_from_str, count_bit_vectors


class rv_bit_vector(object):
    """ A bit vector random variable with a generic distribution.
    """
    
    def __init__(self, xk, pk):
        """ Create a bit vector RV with the specified distribution.

        Parameters:
        -----------
        xk : sequence of bit vectors
        pk : sequence of floats
            A sequence of bit vectors with corresponding probabilities. The bit
            vectors must be of the same length l. Any bit vector of length l not
            specified has zero probability.
        """
        xk = np.array(xk, copy=0)
        pk = np.array(pk, copy=0)
        is_nonzero_p = (pk != 0)
        xk, pk = xk[is_nonzero_p], pk[is_nonzero_p]

        self.length = xk.shape[1]
        self._table = dict(izip(bit_vector_to_str(xk), pk))
        self._xk = xk
        self._rv = rv_discrete(values=(np.arange(len(xk)), pk))

    @classmethod
    def from_samples(cls, d):
        """ Create an estimated RV from a sequence of samples.
        """
        xk, counts = count_bit_vectors(d)
        pk = counts / float(d.shape[0])
        return cls(xk, pk)

    def pmf(self, xk):
        """ The probability mass function evaluated for the given bit vector(s).
        """
        return self._pmf(bit_vector_to_str(xk))

    def _pmf(self, keys):
        return np.array([ self._table.get(k, 0.0) for k in keys ])

    @property
    def support(self):
        """ The support of the distribution.

        Returns:
        --------
        The pair of arrays (bit vectors, probabilities).
        """
        return self._xk, self._rv.pk

    def joint_support(self, other):
        """ The bit vectors that are supported on this or another distribution.

        Returns:
        --------
        The triple of arrays (bit vectors, this probabilities, other RV's
        probabilities).
        """
        keys = list(set(chain(self._table.iterkeys(), 
                              other._table.iterkeys())))
        return (bit_vector_from_str(keys), self._pmf(keys), other._pmf(keys))

    def rvs(self, size=None):
        """ Sample the probability distribution.
        """
        return self._xk[self._rv.rvs(size=size)]


def kl_divergence(p_rv, q_rv, epsilon=1e-6):
    """ Computes the Kullback-Leibler divergence D(p||q) between the
    distributions p and q. 

    Parameters:
    -----------
    p_rv, q_rv : rv_bit_vector
        Bit vector random variables.

    epsilon : float, optional
        Probabilities in q that are less than 'epsilon' are replaced by
        'epsilon' to prevent the KL divergence from blowing up.
    """
    _, p, q = p_rv.joint_support(q_rv)

    # Set p log p = 0 when p == 0.
    is_nonzero_p = (p != 0)
    p = p[is_nonzero_p]
    q = q[is_nonzero_p]

    q = np.maximum(q, epsilon)
    return np.sum(p * np.log(p/q))


def sample_indicator(p, out=None):
    """ Yields 1 with probability p and 0 with probability 1-p. """
    p = np.array(p, copy=0)
    if out is None:
        out = np.ndarray(p.shape, dtype=float)
    return np.less(np.random.sample(p.shape), p, out)
