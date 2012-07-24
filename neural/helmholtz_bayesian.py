# Local imports.
from helmholtz_laddered import LadderedHelmholtzMachine


class BayesianHelmholtzMachine(LadderedHelmholtzMachine):
    """ A laddered Helmholtz machine that performs Bayesian inference.

    The standard Helmholtz machine performs maximum likelihood (ML) learning on
    its parameters. This machine performs maximum a posteriori (MAP) learning on
    its parameters using a prior distribution over the parameter space. The
    techinique offers a principled approach for introducing regularization to
    the wake-sleep learning algorithm.
    """

    # HelmholtzMachine interface

    def __init__(self, topology, ladder_len=None, **kwds):
        """ Create a Bayesian Helmholtz machine.
        """
        super(BayesianHelmholtzMachine, self).__init__(
            topology, ladder_len, **kwds)

        var_0 = 4.0
        top_mean, top_var = self._create_lateral_prior(topology[0], var_0)
        self.G_mean, self.G_var = [ top_mean ], [ top_var ]

    def _wake(self, world, epsilon):
        """ Run a wake cycle.
        """
        return _wake(world, self.G, self.G_lateral, self.G_mean, self.G_var, 
                     self.R, self.R_lateral, epsilon)

    def _sleep(self, epsilon):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_lateral, self.G_mean, self.G_var,
                      self.R, self.R_lateral, epsilon)

    # BayesianHelmholtzMachine interface
    
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
