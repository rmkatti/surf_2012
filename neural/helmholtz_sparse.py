# System library imports.
import numpy as np

# Local imports.
from helmholtz_bayesian import BayesianHelmholtzMachine
from utils.math import logit, sigmoid


class SparseHelmholtzMachine(BayesianHelmholtzMachine):
    """ An HM that forms sparse representations in its hidden layers.
    """

    # BayesianHelmholtzMachine interface
    
    def _create_priors(self):
        """ Create the layer and lateral weight prior hyperparameters.
        """
        layer_params = []
        lateral_params = [ self._create_top_prior(self.topology[0], 
                                                  self.G_ladder_len[0]) ]

        prev_layer_len = self.G_ladder_len[0]
        for layer_len, ladder_len in zip(self.topology, self.G_ladder_len)[1:]:
            layer_param = np.zeros((2, prev_layer_len, layer_len))
            layer_mean, layer_var = layer_param
            layer_var[:,:] = 5.0
            layer_params.append(layer_param)
            
            lateral_param = self._create_lateral_prior(layer_len, ladder_len)
            lateral_params.append(lateral_param)
            
            prev_layer_len = layer_len

        return layer_params, lateral_params

    # SparseHelmholtzMachine interface

    def top_conditional_probs(self):
        """ For each unit in the top layer, compute the probability that it's on
        given that its parents are off.
        """
        top_bias = sigmoid(self.G_lateral[0][:,0])
        probs = np.empty(top_bias.shape)
        probs[0] = top_bias[0]
        probs[1:] = np.cumprod(1-top_bias)[:-1] * top_bias[1:]
        return probs

    def _create_lateral_prior(self, layer_len, ladder_len):
        def fill(x):
            reps = (layer_len + x.shape[0] - 1) / x.shape[0]
            return np.tile(x, (reps, 1))[:layer_len]

        ladder_param = self._create_ladder_prior(ladder_len)
        return np.array(map(fill, ladder_param))
        
    def _create_ladder_prior(self, ladder_len):
        param = np.zeros((2, ladder_len, ladder_len))
        mean, var = param
        lateral_mean = logit(1e-4)
        var[:,0] = 1.0
        for i in xrange(1, ladder_len):
            mean[i,1:i+1] = lateral_mean
            var[i,1:i+1] = 1e-4
        return param

    def _create_top_prior(self, layer_len, ladder_len):
        param = np.zeros((2, layer_len, ladder_len))
        mean, var = param

        # Make top-level units mutually exclusive and equally probable.
        probs = np.reciprocal(np.arange(layer_len, 0, -1, dtype=float))
        probs[-1] -= 1e-3
        bias_mean = logit(probs)
        lateral_mean = logit(1e-6)

        # Make mutual exclusivity high probable but individual probabilities
        # fairly variable.
        mean[:,0] = bias_mean
        var[:-1,0] = 1.0
        var[-1,0] = 1e-4 #4.0 / 80000 #1.0
        for i in xrange(1, layer_len):
            mean[i,1:i+1] = lateral_mean
            var[i,1:i+1] = 1e-4 #0.25 * 12 / 80000 #1e-4

        return param
