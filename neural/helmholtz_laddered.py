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
    
    def __init__(self, topology, ladder_len=5):
        """ Create a laddered Helmholtz machine.
        """
        super(LadderedHelmholtzMachine, self).__init__(topology)

        if np.isscalar(ladder_len):
            self.G_ladder_len = np.repeat(ladder_len, len(topology))
        elif len(ladder_len) == len(topology):
            self.G_ladder_len = np.ndarray(ladder_len)
        else:
            raise ValueError("'topology' and 'ladder_len' have unequal length")
        self.R_ladder_len = self.G_ladder_len[::-1]

        self.G_lateral = self._create_lateral_weights(topology[1:-1]) + [None]
        self.G_bias_lateral = np.zeros((topology[0]-1, topology[0]-1))
        self.R_lateral = self._create_lateral_weights(topology[-2::-1])

    def sample_generative_dist(self, size = None, 
                               all_layers = False, top_units = None):
        """ Sample the generative distribution.
        """
        d = self.G_bias if top_units is None else top_units
        if size is not None:
            d = np.tile(d, (size,1))
        if top_units is None:
            d = _sample_laddered_layer(self.G_bias_lateral,
                                       self.G_ladder_len[0], d)
        samples = _sample_laddered_network(self.G, self.G_lateral,
                                           self.G_ladder_len[1:], d)
        return samples if all_layers else samples[-1]

    def sample_recognition_dist(self, d, size = None):
        """ Sample the recognition distribution for the given data.
        """
        if size is not None:
            d = np.tile(d, (size,1))
        samples = _sample_laddered_network(self.R, self.R_lateral,
                                           self.R_ladder_len[1:], d)
        samples.reverse()
        return samples

    def _generative_probs_for_sample(self, samples):
        """ The generative probabilities for each unit in the network, given a
        sample of the hidden units.
        """
        probs = _probs_for_laddered_network(self.G, self.G_lateral,
                                            self.G_ladder_len[1:], samples)
        s = samples[0]
        p_in = np.tile(self.G_bias, s.shape[:-1] + (1,))
        p = _probs_for_laddered_layer(self.G_bias_lateral,
                                      self.G_ladder_len[0], p_in, s)
        probs.insert(0, p)
        return probs

    def _wake(self, world, epsilon):
        """ Run a wake cycle.
        """
        return _wake(world, self.G, self.G_bias, self.R,
                     self.G_lateral, self.G_bias_lateral, self.R_lateral,
                     self.G_ladder_len, self.R_ladder_len, epsilon)

    def _sleep(self, epsilon):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_bias, self.R,
                      self.G_lateral, self.G_bias_lateral, self.R_lateral,
                      self.G_ladder_len, self.R_ladder_len, epsilon)

    # LadderedHelmholtzMachine interface

    def _create_lateral_weights(self, topology):
        """ Create a list of lateral connection weight matrices for the given
        network topology.
        """
        return [ np.zeros((layer-1, layer-1)) for layer in topology ]

# Laddered Helmholtz machine internals

def _sample_laddered_layer(lateral, ladder_len, s_in):
    if lateral is None:
        return sample_indicator(sigmoid(s_in))
    s = np.empty(s_in.shape)
    s[...,0] = sample_indicator(sigmoid(s_in[...,0]))
    for i in range(1, s.shape[-1]):
        start = max(0, i - ladder_len)
        s_in[...,i] += s[...,start:i].dot(lateral[i-1,start:i])
        s[...,i] = sample_indicator(sigmoid(s_in[...,i]))
    return s

def _sample_laddered_network(layers, laterals, ladder_lens, s):
    samples = [ s ]
    for layer, lateral, ladder_len in zip(layers, laterals, ladder_lens):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = _sample_laddered_layer(lateral, ladder_len, s_in)
        samples.append(s)
    return samples

def _probs_for_laddered_layer(lateral, ladder_len, p_in, s):
    if lateral is not None:
        for i in range(1, s.shape[-1]):
            start = max(0, i - ladder_len)
            p_in[...,i] += s[...,start:i].dot(lateral[i-1,start:i])
    return sigmoid(p_in)

def _probs_for_laddered_network(layers, laterals, ladder_lens, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, ladder_len, s in \
            zip(layers, laterals, ladder_lens, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        p = _probs_for_laddered_layer(lateral, ladder_len, p_in, s)
        probs.append(p)
        s_prev = s
    return probs

from _helmholtz_laddered import _wake, _sleep
