# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from util import sample_indicator, sigmoid

# Helmholtz machine public API

class HelmholtzMachine(object):
    """ A classic Helmholtz machine.
    """
    
    def __init__(self, topology):
        """ Create a Helmholtz machine.

        Parameters:
        -----------
        topology : sequence of ints
            The topology of the network. Gives the node count for each layer in
            "top-to-bottom" order. That is, the head of the sequence specifies
            the generative bias nodes, while the last element of the sequence
            specifies the input nodes.
        """
        self.topology = topology
        self.G = self._create_layer_weights(topology)
        self.G_bias = np.zeros(topology[0])
        self.R = self._create_layer_weights(reversed(topology))

    def train(self, world, epsilon=None, maxiter=None, 
              yield_at=None, yield_call=None):
        """ Train the Helmholtz machine.

        Parameters:
        -----------
        world : callable() -> bit vector
            A function that samples data from the "world".

            Note: the number of input nodes in the network topology must equal
            the size of the bit vectors produced by the 'world' function.

        epsilon : float or sequence of floats, optional [default 0.01]
            The step size for the weight update rules. If a sequence is given,
            it must be of the same size as 'topology'; a different step size may
            then be used at each layer.

        maxiter : int, optional [default 50000]
            The number the wake-sleep cycles to run.

        yield_at: int, optional [default 1000]
        yield_call : callable(iter), optional
            If provided, the given function will be called periodically with the
            current iteration number.
        """
        epsilon = epsilon or 0.01
        if np.isscalar(epsilon):
            epsilon = np.repeat(epsilon, len(self.topology))
        else:
            epsilon = np.array(epsilon, copy=0)
        maxiter = maxiter or 50000
        yield_at = yield_at or 1000

        if yield_call:
            next_yield = yield_at
            yield_call(0)

        for i in xrange(1, maxiter+1):
            self._wake(world, epsilon)
            self._sleep(epsilon)
            if yield_call and next_yield == i:
                next_yield += yield_at
                yield_call(i)

    def estimate_coding_cost(self, d, n = 10):
        """ Estimate the expected coding cost of the data d by sampling the
        recognition distribution.
        """
        samples = self.sample_recognition_dist(d, size=n)
        probs = self._generative_probs_for_sample(samples)
        costs = np.zeros(n)
        for s, prob in izip(samples, probs):
            costs += np.sum(-s*np.log(prob) - (1-s)*np.log(1-prob), axis=1)
        return costs.mean()
        
    def estimate_generative_dist(self, n = 10000):
        """ Estimate the generative distribution by sampling.
        """
        from prob import rv_bit_vector
        d = self.sample_generative_dist(size=n)
        return rv_bit_vector.from_samples(d)

    def sample_generative_dist(self, size = None, 
                               all_layers = False, top_units = None):
        """ Sample the generative distribution.

        Parameters:
        -----------
        size : int, optional [default None]
            The number of samples to draw. If None, returns a single sample.

        all_layers : bool, optional [default False]
            By default, an array of input unit samples is returned. If
            'all_layers` is True, a list of sample arrays for *all* the layers,
            in top-to-bottom order, is returned.

        top_units : bit vector, optional
            By default, the top-level units are sampled from the generative
            biases. This parameter clamps the top-level units to specific
            values.

        Returns:
        --------
        A (list of) 2D sample array(s), where the first dimension indexes the
        individual samples. See 'all_layers' parameter.
        """
        d = self.G_bias if top_units is None else top_units
        if size is not None:
            d = np.tile(d, (size,1))
        if top_units is None:
            d = sample_indicator(sigmoid(d))
        samples = _sample_layered_network(self.G, d)
        return samples if all_layers else samples[-1]

    def sample_recognition_dist(self, d, size = None):
        """ Sample the recognition distribution for the given data.

        Parameters:
        -----------
        d : bit vector
            The data to input to the recognition model.

        size : int, optional [default None]
            The number of samples to draw. If None, returns a single sample.

        Returns:
        --------
        A list of 2D sample arrays for the network layers, in top-to-bottom
        order, including the input units. The first dimension indexes the
        individual samples.
        """
        if size is not None:
            d = np.tile(d, (size,1))
        samples = _sample_layered_network(self.R, d)
        samples.reverse()
        return samples

    def _create_layer_weights(self, topology):
        """ Create a list of inter-layer weight matrices for the given network
        topology.
        """
        weights = []
        topology = tuple(topology)
        for top, bottom in izip(topology, topology[1:]):
            weights.append(np.zeros((top + 1, bottom)))
        return weights

    def _generative_probs_for_sample(self, samples):
        """ The generative probabilities for each unit in the network, given a
        sample of the hidden units.
        """
        probs = _probs_for_layered_network(self.G, samples)
        probs.insert(0, sigmoid(self.G_bias))
        return probs

    def _wake(self, world, epsilon):
        """ Run a wake cycle.
        """
        return _wake(world, self.G, self.G_bias, self.R, epsilon)

    def _sleep(self, epsilon):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_bias, self.R, epsilon)

# Helmholtz machine internals

def _sample_layered_network(layers, s):
    samples = [ s ]
    for L in layers:
        s = sample_indicator(sigmoid(s.dot(L[:-1]) + L[-1]))
        samples.append(s)
    return samples

def _probs_for_layered_network(layers, samples):
    return [ sigmoid(s.dot(L[:-1]) + L[-1])
             for L, s in izip(layers, samples) ]
        
# Reference/fallback implementation
def _wake(world, G, G_bias, R, epsilon):
    # Sample data from the world.
    s = world()
    samples = _sample_layered_network(R, s)
    samples.reverse()
        
    # Pass back down through the generation network and adjust weights.
    G_bias += epsilon[0] * (samples[0] - sigmoid(G_bias))
    G_probs = _probs_for_layered_network(G, samples)
    for G_weights, inputs, target, generated, step \
            in izip(G, samples, samples[1:], G_probs, epsilon[1:]):
        G_weights[:-1] += step * np.outer(inputs, target - generated)
        G_weights[-1] += step * (target - generated)

# Reference/fallback implementation
def _sleep(G, G_bias, R, epsilon):
    # Generate a dream.
    d = sample_indicator(sigmoid(G_bias))
    dreams = _sample_layered_network(G, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_layered_network(R, dreams)
    for R_weights, inputs, target, recognized, step \
            in izip(R, dreams, dreams[1:], R_probs, epsilon[::-1]):
        R_weights[:-1] += step * np.outer(inputs, target - recognized)
        R_weights[-1] += step * (target - recognized)

try:
    from _helmholtz import _wake, _sleep
except ImportError:
    print 'WARNING: Optimized HM not available.'
