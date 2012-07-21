# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from helmholtz import HelmholtzMachine
from utils.math import sample_indicator, sigmoid

# Laddered Helmholtz machine public API

class LadderedHelmholtzMachine(HelmholtzMachine):
    """ A Helmholtz machine with
    """

    # HelmholtzMachine interface
    
    def __init__(self, topology, ladder_len=None):
        """ Create a laddered Helmholtz machine.

        Parameters:
        -----------
        topology : sequence of ints
            See HelmholtzMachine.

        ladder_len : int or sequence of ints, optional
            The (maximum) length of the ladders for the nodes in each layer, in
            top-to-bottom order. If not specified, each hidden layer is fully
            laterally connected, while the visible layer has no lateral
            connections (in the generative model).
        """
        super(LadderedHelmholtzMachine, self).__init__(topology)

        if ladder_len is None:
            self.G_ladder_len = np.array(list(topology[:-1]) + [0])
        elif np.isscalar(ladder_len):
            self.G_ladder_len = np.repeat(ladder_len, len(topology))
        elif len(ladder_len) == len(topology):
            self.G_ladder_len = np.ndarray(ladder_len)
        else:
            raise ValueError("'topology' and 'ladder_len' have unequal length")
        self.R_ladder_len = self.G_ladder_len[-2::-1]
        
        G_laterals = self._create_lateral_weights(topology, self.G_ladder_len)
        self.G_bias_lateral, self.G_lateral = G_laterals[0], G_laterals[1:]
        self.R_lateral = self._create_lateral_weights(topology[-2::-1],
                                                      self.R_ladder_len)

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
                                           self.R_ladder_len, d)
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

    def _create_lateral_weights(self, topology, lateral_lens):
        """ Create a list of lateral connection weight matrices for the given
        network topology.
        """
        laterals = []
        for layer, lateral_len in izip(topology, lateral_lens):
            lateral = np.zeros((layer-1, layer-1)) if lateral_len else None
            laterals.append(lateral)
        return laterals

# Laddered Helmholtz machine internals

def _sample_laddered_layer(lateral, ladder_len, s_in):
    if ladder_len == 0:
        return sample_indicator(sigmoid(s_in))

    s = s_in.copy()
    s[...,0] = sample_indicator(sigmoid(s[...,0]))
    for i in range(1, s.shape[-1]):
        start = max(0, i - ladder_len)
        s[...,i] += s[...,start:i].dot(lateral[i-1,start:i])
        s[...,i] = sample_indicator(sigmoid(s[...,i]))
    return s

def _sample_laddered_network(layers, laterals, ladder_lens, s):
    samples = [ s ]
    for layer, lateral, ladder_len in izip(layers, laterals, ladder_lens):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = _sample_laddered_layer(lateral, ladder_len, s_in)
        samples.append(s)
    return samples

def _probs_for_laddered_layer(lateral, ladder_len, p_in, s):
    if ladder_len == 0:
        return sigmoid(p_in)

    p = p_in.copy()
    for i in range(1, s.shape[-1]):
        start = max(0, i - ladder_len)
        p[...,i] += s[...,start:i].dot(lateral[i-1,start:i])
    return sigmoid(p, out=p)

def _probs_for_laddered_network(layers, laterals, ladder_lens, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, ladder_len, s in \
            izip(layers, laterals, ladder_lens, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        p = _probs_for_laddered_layer(lateral, ladder_len, p_in, s)
        probs.append(p)
        s_prev = s
    return probs

from _helmholtz_laddered import _wake, _sleep
