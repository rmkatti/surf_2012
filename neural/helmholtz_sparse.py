# System library imports.
import numpy as np

# Local imports.
from helmholtz_bayesian import BayesianHelmholtzMachine


class SparseHelmholtzMachine(BayesianHelmholtzMachine):
    """
    """

    def _create_lateral_prior(self, n, var_0):
        """
        """
        i = np.arange(1, n+1, dtype=float)
        probs = np.reciprocal(i[::-1])
        probs[-1] -= 1e-4
        bias_mean = logit(probs)
        
        #lateral_mean = -(bias_mean[1:] - bias_mean[0]) * n / (i[1:]-1)
        lateral_mean = np.repeat(logit(1e-4), n-1)

        mean = np.zeros((n, n))
        mean[:,0] = bias_mean
        for k in xrange(1, n):
            mean[k,1:k+1] = lateral_mean[k-1]
            
        #u = np.insert(lateral_mean, 0, 0)
        #var = (n/(n+i-1)) * (var_0 - (n-i+1)*(i-1) * u / n**2)
        var = np.array([4.0] + [0.25] * (n-1)) * 12
        var = var.reshape((n,1))
        
        return mean, var
