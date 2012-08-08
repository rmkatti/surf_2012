# System library imports.
import numpy as np

# Local imports.
from helmholtz_laddered import LadderedHelmholtzMachine


class BayesianHelmholtzMachine(LadderedHelmholtzMachine):
    """ A laddered Helmholtz machine that performs Bayesian inference.

    The standard Helmholtz machine performs maximum likelihood (ML) learning on
    its parameters. This machine performs maximum a posteriori (MAP) learning on
    its parameters using a prior distribution over the parameter space. The
    techinique offers a principled approach for introducing regularization to
    the wake-sleep learning algorithm.

    This class assumes that each parameter is independently normally distributed
    under the prior. In the limit of the variances approaching infinity, the
    algorithm reduces to ML learning.
    """

    # HelmholtzMachine interface

    def __init__(self, topology, ladder_len=None, **params):
        """ Create a Bayesian Helmholtz machine.
        """
        super(BayesianHelmholtzMachine, self).__init__(topology, ladder_len)
        self.G_param, self.G_lateral_param = self._create_priors(**params)
        for layer, (mean, _) in zip(self.G, self.G_param):
            layer[:,:] = mean
        for lateral, (mean, _) in zip(self.G_lateral, self.G_lateral_param):
            lateral[:,:] = mean

    def _wake(self, sample, data_size, iteration, rate):
        """ Run a wake cycle.
        """
        return _wake(sample, self.G, self.G_param,
                     self.G_lateral, self.G_lateral_param,
                     self.R, self.R_lateral, data_size, rate)

    # BayesianHelmholtzMachine interface
    
    def _create_priors(self, base_variance = 4.0):
        """ Create the layer and lateral weight prior hyperparameters.

        The default implementation establishes the unique prior satisfying:

            1. Each unit is independent and equally likely to have each of the
               values 0 and 1.
            2. The mean of each paramater is 0.
            3. All parameters for a particular unit have the same variance.
            4. The variance of the input to each unit is constant under the
               prior and equal to 'base_variance'.

        See Pearl, "Graphical models for machine learning", Section 3.2.3.
        """
        self.base_variance = base_variance
        layer_params, lateral_params = [], []

        prev_layer_len = 0
        for layer_len, ladder_len in zip(self.topology, self.G_ladder_len):
            # Count the number of incoming connections to each unit.
            count = np.minimum(ladder_len, np.arange(1, layer_len+1)) # lateral
            count += prev_layer_len # inter-layer

            # Compute the parameter variance for each unit.
            var = (2.0 / (count + 1)) * base_variance

            if prev_layer_len:
                layer_param = np.zeros((2, prev_layer_len, layer_len))
                layer_mean, layer_var = layer_param
                layer_var[:,:] = var
                layer_params.append(layer_param)

            lateral_param = np.zeros((2, layer_len, ladder_len))
            lateral_mean, lateral_var = lateral_param
            for i in xrange(layer_len):
                lateral_var[i,:min(i+1, ladder_len)] = var[i]
            lateral_params.append(lateral_param)

            prev_layer_len = layer_len

        return layer_params, lateral_params
        

from _helmholtz_bayesian import _wake
