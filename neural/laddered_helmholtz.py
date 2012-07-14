# Standard library imports.
from itertools import izip

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

def _sample_laddered_bias(bias, lateral):
    s = np.empty(bias.shape)
    s_in = bias.copy()
    s[...,0] = sample_indicator(sigmoid(s_in[...,0]))
    for i in xrange(1, len(lateral)+1):
        s_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
        s[...,i] = sample_indicator(sigmoid(s_in[...,i]))
    return s

def _sample_laddered_network(layers, laterals, s):
    samples = [ s ]
    for layer, lateral in izip(layers, laterals):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = np.empty(s_in.shape)
        s[...,0] = sample_indicator(sigmoid(s_in[...,0]))
        for i in xrange(1, len(lateral)+1):
            s_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
            s[...,i] = sample_indicator(sigmoid(s_in[...,i]))
        samples.append(s)
    return samples

def _probs_for_laddered_bias(bias, lateral, s):
    p_in = bias.copy()
    for i in xrange(1, len(lateral)+1):
        p_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
    return sigmoid(p_in)

def _probs_for_laddered_network(layers, laterals, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, s in izip(layers, laterals, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        for i in xrange(1, len(lateral)+1):
            p_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
        p = sigmoid(p_in)
        s_prev = s
        probs.append(p)
    return probs
        
def _wake(world, G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    # Sample data from the world.
    s = world()
    samples = _sample_laddered_network(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    generated = _probs_for_laddered_bias(G_bias, G_bias_lateral, samples[0])
    G_bias += epsilon[0] * (samples[0] - generated)
    G_probs = _probs_for_laddered_network(G, G_lateral, samples)
    for layer, lateral, inputs, target, generated, step \
            in izip(G, G_lateral, samples, samples[1:], G_probs, epsilon[1:]):
        layer[:-1] += step * np.outer(inputs, target - generated)
        layer[-1] += step * (target - generated)
        lateral += step * np.outer(target[1:] - generated[1:], target[1:])

def _sleep(G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    # Generate a dream.
    d = _sample_laddered_bias(G_bias, G_bias_lateral)
    dreams = _sample_laddered_network(G, G_lateral, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in izip(R, R_lateral, dreams, dreams[1:], R_probs, epsilon[::-1]):
        layer[:-1] += step * np.outer(inputs, target - recognized)
        layer[-1] += step * (target - recognized)
        lateral += step * np.outer(target[1:] - recognized[1:], target[1:])
