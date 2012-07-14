# System library imports.
import numpy as np

# Local imports.
from helmholtz import HelmholtzMachine
from util import sample_indicator, sigmoid

# Laddered Helmholtz machine public API

class LadderedHelmholtzMachine(HelmholtzMachine):
    """ A Helmholtz machine with
    """

    # HelmholtzMachine interface
    
    def __init__(self, topology):
        """ Create a laddered Helmholtz machine.
        """
        super(LadderedHelmholtzMachine, self).__init__(topology)
        self.G_lateral = self._create_lateral_weights(topology)
        self.G_bias_lateral = np.zeros((topology[0]-1, topology[0]-1))
        self.R_lateral = self._create_lateral_weights(reversed(topology))

    def sample_generative_dist(self, size = None, 
                               all_layers = False, top_units = None):
        """ Sample the generative distribution.
        """
        d = self.G_bias if top_units is None else top_units
        if size is not None:
            d = np.tile(d, (size,1))
        if top_units is None:
            d = sample_indicator(sigmoid(d))
        samples = _sample_laddered_network(self.G, self.G_lateral, d)
        return samples if all_layers else samples[-1]

    def sample_recognition_dist(self, d, size = None):
        """ Sample the recognition distribution for the given data.
        """
        if size is not None:
            d = np.tile(d, (size,1))
        samples = _sample_laddered_network(self.R, self.R_lateral, d)
        samples.reverse()
        return samples

    def _generative_probs_for_sample(self, samples):
        """ The generative probabilities for each unit in the network, given a
        sample of the hidden units.
        """
        probs = _probs_for_laddered_network(self.G, self.G_lateral, samples)
        probs.insert(0, _probs_for_laddered_bias(
                self.G_bias, self.G_lateral_bias, samples[0]))
        return probs

    def _wake(self, world, epsilon):
        """ Run a wake cycle.
        """
        return _wake(world, self.G, self.G_bias, self.R, self.G_lateral, 
                     self.G_bias_lateral, self.R_lateral, epsilon)

    def _sleep(self, epsilon):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_bias, self.R, self.G_lateral, 
                      self.G_bias_lateral, self.R_lateral, epsilon)

    # LadderedHelmholtzMachine interface

    def _create_lateral_weights(self, topology):
        """ Create a list of lateral connection weight matrices for the given
        network topology.
        """
        topology = tuple(topology)
        return [ np.zeros((layer-1, layer-1)) for layer in topology[1:] ]

# Laddered Helmholtz machine internals

from _helmholtz_laddered import \
    _sample_laddered_bias, _sample_laddered_network, \
    _probs_for_laddered_bias, _probs_for_laddered_network, _wake, _sleep
