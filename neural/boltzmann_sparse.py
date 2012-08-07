# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from helmholtz import HelmholtzMachine
from utils.math import sample_exclusive_indicators, sample_indicator, sigmoid

# Sparse Boltzmann machine public API

class SparseBoltzmannMachine(HelmholtzMachine):
    """ A cross between a Boltzmann machine and a Helmholtz machine.

    The network is multi-layered like the Helmholtz machine and learns using the
    wake-sleep algorithm. Individual layers, however, have undirected lateral
    connections, as in the Boltzmann machine. Inference is made tractable by
    enforcing a strict sparsity assumption.
    """

    # HelmholtzMachine interface

    def __init__(self, topology, group_sizes=None):
        """ Create a sparse Boltzmann machine.

        Parameters:
        -----------
        topology : sequence of ints
            See HelmholtzMachine machine.

        group_sizes : sequence of ints
            Specifies the lateral connectivity of each layer, in top to bottom
            order. The length of the sequence must equal that of 'topology'.

            Each layer is divided evenly into groups of the given size. The
            units in each group are mutually exclusive, while the group
            themselves are conditionally independent given the previous layer.
            In the degenerate case where the group size is 1, the layer is
            completely factorial, as in the Helmholtz machine. The group size
            for the input layer should be 1.

            By default, the top layer is a single group, while all other layers
            have group size 1.
        """
        super(SparseBoltzmannMachine, self).__init__(topology)
        if group_sizes is None:
            group_sizes = [topology[0]] + [1] * (len(topology)-1)
        elif np.any(np.mod(topology, group_sizes)):
            raise ValueError("group sizes must divide layer sizes")
        self.group_sizes = list(group_sizes)

    def sample_generative_dist(self, size = None, 
                               all_layers = False, top_units = None):
        """ Sample the generative distribution.
        """
        if top_units is None:
            if size is None:
                d = self.G_top.copy()
            else:
                d = np.tile(self.G_top, (size,1))
            _sample_boltzmann_layer(self.group_sizes[0], d)
        else:
            d = top_units if size is None else np.tile(top_units, (size,1))
        samples = _sample_boltzmann_network(self.G, self.group_sizes[1:], d)
        return samples if all_layers else samples[-1]

    def sample_recognition_dist(self, d, size = None):
        """ Sample the recognition distribution for the given data.
        """
        if size is not None:
            d = np.tile(d, (size,1))
        samples = _sample_boltzmann_network(self.R, self.group_sizes[-2::-1], d)
        samples.reverse()
        return samples

    def _generative_probs_for_sample(self, samples):
        """ The generative probabilities for each unit in the network, given a
        sample of the hidden units.
        """
        probs = _probs_for_boltzmann_network(
            self.G, self.group_sizes[1:], samples)
        p = _probs_for_boltzmann_layer(self.group_sizes[0], self.G_top.copy())
        probs.insert(0, p)
        return probs

    def _wake(self, sample, data_size, epochs, rate):
        """ Run a wake cycle.
        """
        return _wake(sample, self.group_sizes, self.G, self.G_top, self.R, rate)

    def _sleep(self, data_size, epochs, rate):
        """ Run a sleep cycle.
        """
        return _sleep(self.group_sizes, self.G, self.G_top, self.R, rate)

# Sparse Boltzmann machine internals

def _sample_boltzmann_layer(group_size, s):
    # Shortcut for degenerate case.
    if group_size == 1:
        return sample_indicator(sigmoid(s, out=s), out=s)

    s.shape = s.shape[:-1] + (-1, group_size)
    boltzmann_dist(s, axis=-1, out=s)
    sample_exclusive_indicators(s, axis=-1, out=s)
    s.shape = s.shape[:-2] + (-1,)
    return s

def _sample_boltzmann_network(layers, group_sizes, s):
    samples = [ s ]
    for layer, group_size in izip(layers, group_sizes):
        s = s.dot(layer[:-1]) + layer[-1]
        _sample_boltzmann_layer(group_size, s)
        samples.append(s)
    return samples

def _probs_for_boltzmann_layer(group_size, p):
    # Shortcut for degenerate case.
    if group_size == 1:
        return sigmoid(p, out=p)
    
    p.shape = p.shape[:-1] + (-1, group_size)
    boltzmann_dist(p, axis=-1, out=p)
    p.shape = p.shape[:-2] + (-1,)
    return p

def _probs_for_boltzmann_network(layers, group_sizes, samples):
    probs = []
    for layer, group_size, s in izip(layers, group_sizes, samples):
        p = s.dot(layer[:-1]) + layer[-1]
        _probs_for_boltzmann_layer(group_size, p)
        probs.append(p)
    return probs

def _wake(sample, group_sizes, G, G_top, R, rate):
    # Sample data from the recognition network.
    samples = _sample_boltzmann_network(R, group_sizes[-2::-1], sample)
    samples.reverse()
        
    # Pass back down through the generation network and adjust weights.
    generated = G_top.copy()
    _probs_for_boltzmann_layer(group_sizes[0], generated)
    G_top += rate[0] * (samples[0] - generated)

    G_probs = _probs_for_boltzmann_network(G, group_sizes[1:], samples)
    for G_weights, inputs, target, generated, step \
            in izip(G, samples, samples[1:], G_probs, rate[1:]):
        G_weights[:-1] += step * np.outer(inputs, target - generated)
        G_weights[-1] += step * (target - generated)

def _sleep(group_sizes, G, G_top, R, rate):
    # Generate a dream.
    d = G_top.copy()
    _sample_boltzmann_layer(group_sizes[0], d)
    dreams = _sample_boltzmann_network(G, group_sizes[1:], d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_boltzmann_network(R, group_sizes[-2::-1], dreams)
    for R_weights, inputs, target, recognized, step \
            in izip(R, dreams, dreams[1:], R_probs, rate[::-1]):
        R_weights[:-1] += step * np.outer(inputs, target - recognized)
        R_weights[-1] += step * (target - recognized)

try:
    from _boltzmann_sparse import boltzmann_dist
except ImportError:
    print 'WARNING: Optimized sparse Boltzmann machine not available.'

    def boltzmann_dist(x, axis=-1, out=None):
        """ Compute the Boltzmann distribution (at fundamental temperature 1)
        for an array of energies. An extra 'zero-point' energy is included.
        """
        if out is None:
            out = np.empty(x.shape, dtype=np.double)
        exp_x = np.exp(x, out=out)
        Z = np.expand_dims(1.0 + exp_x.sum(axis=axis), axis)
        return np.divide(exp_x, Z, out=out)
