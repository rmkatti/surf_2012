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

            The minimum ladder length is 1, corresponding to a constant bias and
            equivalent to the factorial Helmholtz machine.
        """
        self.topology = list(topology)
        self.G = self._create_layer_weights(topology, biases=False)
        self.R = self._create_layer_weights(topology[::-1], biases=False)

        if ladder_len is None:
            self.G_ladder_len = np.array(list(topology[:-1]) + [1])
        elif np.isscalar(ladder_len):
            self.G_ladder_len = np.repeat(ladder_len, len(topology))
        elif len(ladder_len) == len(topology):
            self.G_ladder_len = np.array(ladder_len)
        else:
            raise ValueError("'topology' and 'ladder_len' have unequal length")
        if not np.all(self.G_ladder_len >= 1):
            raise ValueError("'ladder_len' must be at least 1")
        self.R_ladder_len = self.G_ladder_len[-2::-1]

        self.G_lateral = self._create_lateral_weights(topology,
                                                      self.G_ladder_len)
        self.R_lateral = self._create_lateral_weights(topology[-2::-1],
                                                      self.R_ladder_len)

    def sample_generative_dist(self, size = None, 
                               all_layers = False, top_units = None):
        """ Sample the generative distribution.
        """
        if top_units is None:
            width = self.G_lateral[0].shape[0]
            d = np.zeros(width if size is None else (size, width))
            _sample_laddered_layer(self.G_lateral[0], d)
        else:
            d = top_units if size is None else np.tile(top_units, (size,1))
        samples = _sample_laddered_network(self.G, self.G_lateral[1:], d)
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
        probs = _probs_for_laddered_network(self.G, self.G_lateral[1:], samples)
        p = np.zeros(samples[0].shape)
        _probs_for_laddered_layer(self.G_lateral[0], samples[0], p)
        probs.insert(0, p)
        return probs

    def _wake(self, sample, data_size, epochs, rate):
        """ Run a wake cycle.
        """
        return _wake(sample, self.G, self.G_lateral, self.R, self.R_lateral,
                     rate)

    def _sleep(self, data_size, epochs, rate):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_lateral, self.R, self.R_lateral, rate)

    # LadderedHelmholtzMachine interface

    def _create_lateral_weights(self, topology, lateral_lens):
        """ Create a list of lateral connection weight matrices for the given
        network topology.
        """
        return [ np.zeros((layer, lateral_len))
                 for layer, lateral_len in izip(topology, lateral_lens) ]

# Laddered Helmholtz machine internals

def _sample_laddered_layer(lateral, s):
    s[...,:] += lateral[:,0]
    s[...,0] = sample_indicator(sigmoid(s[...,0]))
    for i in range(1, s.shape[-1]):
        j = min(i, lateral.shape[1]-1)
        s[...,i] += s[...,i-j:i].dot(lateral[i,j:0:-1])
        s[...,i] = sample_indicator(sigmoid(s[...,i]))
    return s

def _sample_laddered_network(layers, laterals, s):
    samples = [ s ]
    for layer, lateral, in izip(layers, laterals):
        s = s.dot(layer)
        _sample_laddered_layer(lateral, s)
        samples.append(s)
    return samples

def _probs_for_laddered_layer(lateral, s, p):
    p[...,:] += lateral[:,0]
    for i in range(1, s.shape[-1]):
        j = min(i, lateral.shape[1]-1)
        p[...,i] += s[...,i-j:i].dot(lateral[i,j:0:-1])
    return sigmoid(p, out=p)

def _probs_for_laddered_network(layers, laterals, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, s in izip(layers, laterals, samples[1:]):
        p = s_prev.dot(layer)
        _probs_for_laddered_layer(lateral, s, p)
        probs.append(p)
        s_prev = s
    return probs

from _helmholtz_laddered import _wake, _sleep
