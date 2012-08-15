# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from utils.math import sample_indicator, sigmoid

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
            top-to-bottom order. That is, the head of the sequence specifies the
            generative bias nodes, while the last element of the sequence
            specifies the input nodes.
        """
        self.topology = list(topology)
        self.G = self._create_layer_weights(topology)
        self.G_top = np.zeros(topology[0])
        self.R = self._create_layer_weights(topology[::-1])

    def train(self, data, rate=None, anneal=None, epochs=None,
              yield_at=None, yield_call=None):
        """ Train the Helmholtz machine using the wake-sleep algorihtm.

        Parameters:
        -----------
        data : sequence of bit vector
            The training data. Note that the number of input nodes in the
            network topology must equal the size of the data vectors.

        rate : float or sequence of floats, optional [default 0.01]
            The learning rate, i.e. the step size for the weight updates. If a
            different learning rate is required at each layer, a sequence may be
            given, which must be of the same length as 'topology'.

        anneal : float or sequence of floats, optional
            By default, the learning rate is constant. If 'anneal' is specified,
            the learning rate is decreased according to the schedule

                rate = rate_0 / (1 + aneal * e),
                
            where e is the current epoch.

        epochs : int, optional [default 100]
            The number the epochs (full pases through the data set).

        yield_at: int, optional [default one epoch]
        yield_call : callable(iter), optional
            If provided, the given function will be called periodically with the
            current iteration number (cumulative count of wake-sleep cycles).
        """
        data_size = len(data)

        if rate is None:
            rate = 0.01
        if np.isscalar(rate):
            rate = np.repeat(rate, len(self.topology))
        else:
            rate = np.array(rate, copy=0)

        if anneal:
            rate_0 = rate.copy()
            if not np.isscalar(anneal):
                anneal = np.array(anneal, copy=0)
        
        if yield_call:
            yield_at = yield_at or data_size
            next_yield = yield_at
            yield_call(0)

        iteration = 1
        for epoch in xrange(1, epochs+1):
            indices = np.random.permutation(data_size)
            for index in indices:
                sample = data[index]
                self._wake(sample, data_size, iteration, rate)
                self._sleep(data_size, iteration, rate)

                if yield_call and next_yield == iteration:
                    next_yield += yield_at
                    yield_call(iteration)

                iteration += 1
            if anneal:
                rate = rate_0 / (1.0 + anneal * epoch)

    def estimate_coding_cost(self, d, n = 10):
        """ Estimate the expected coding cost (in bits) of the given data by
        sampling the recognition distribution.
        """
        samples = self.sample_recognition_dist(d, size=n)
        probs = self._generative_probs_for_sample(samples)
        costs = np.zeros(n)
        for s, prob in izip(samples, probs):
            costs += np.sum(-s*np.log2(prob) - (1-s)*np.log2(1-prob), axis=1)
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
        d = self.G_top if top_units is None else top_units
        if size is not None:
            d = np.tile(d, (size,1))
        if top_units is None:
            d = sample_indicator(sigmoid(d))
        samples = _sample_factorial_network(self.G, d)
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
        samples = _sample_factorial_network(self.R, d)
        samples.reverse()
        return samples

    def _create_layer_weights(self, topology, biases=True):
        """ Create a list of inter-layer weight matrices for the given network
        topology.
        """
        weights = []
        for top, bottom in izip(topology, topology[1:]):
            if biases: top += 1
            weights.append(np.zeros((top, bottom)))
        return weights

    def _generative_probs_for_sample(self, samples):
        """ The generative probabilities for each unit in the network, given a
        sample of the hidden units.
        """
        probs = _probs_for_factorial_network(self.G, samples)
        probs.insert(0, sigmoid(self.G_top))
        return probs

    def _wake(self, sample, data_size, iteration, rate):
        """ Run a wake cycle.
        """
        return _wake(sample, self.G, self.G_top, self.R, rate)

    def _sleep(self, data_size, iteration, rate):
        """ Run a sleep cycle.
        """
        return _sleep(self.G, self.G_top, self.R, rate)

# Helmholtz machine internals

def _sample_factorial_network(layers, s):
    samples = [ s ]
    for L in layers:
        s = sample_indicator(sigmoid(s.dot(L[:-1]) + L[-1]))
        samples.append(s)
    return samples

def _probs_for_factorial_network(layers, samples):
    return [ sigmoid(s.dot(L[:-1]) + L[-1])
             for L, s in izip(layers, samples) ]
        
# Reference/fallback implementation
def _wake(sample, G, G_top, R, rate):
    # Sample data from the recognition network.
    samples = _sample_factorial_network(R, sample)
    samples.reverse()
        
    # Pass back down through the generation network and adjust weights.
    G_top += rate[0] * (samples[0] - sigmoid(G_top))
    G_probs = _probs_for_factorial_network(G, samples)
    for G_weights, inputs, target, generated, step \
            in izip(G, samples, samples[1:], G_probs, rate[1:]):
        G_weights[:-1] += step * np.outer(inputs, target - generated)
        G_weights[-1] += step * (target - generated)

# Reference/fallback implementation
def _sleep(G, G_top, R, rate):
    # Generate a dream.
    d = sample_indicator(sigmoid(G_top))
    dreams = _sample_factorial_network(G, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_factorial_network(R, dreams)
    for R_weights, inputs, target, recognized, step \
            in izip(R, dreams, dreams[1:], R_probs, rate[::-1]):
        R_weights[:-1] += step * np.outer(inputs, target - recognized)
        R_weights[-1] += step * (target - recognized)

try:
    from _helmholtz import _wake, _sleep
except ImportError:
    print 'WARNING: Optimized HM not available.'
