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
        self.G = self._create_layered_network(topology)
        self.G_bias = np.zeros(topology[0])
        self.R = self._create_layered_network(reversed(topology))

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
        G, G_bias, R = self.G, self.G_bias, self.R
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
            _wake(world, G, G_bias, R, epsilon)
            _sleep(G, G_bias, R, epsilon)
            if yield_call and next_yield == i:
                next_yield += yield_at
                yield_call(i)

    def estimate_coding_cost(self, d, n = 10):
        """ Estimate the expected coding cost of the data d by sampling the
        recognition distribution.
        """
        G, G_bias = self.G, self.G_bias
        costs = np.zeros(n)
        samples = self.sample_recognition_dist(d, n)

        # Coding cost for hidden units.
        prob = sigmoid(G_bias)
        for G_weights, s in izip(G, samples):
            costs += np.sum(-s*np.log(prob) - (1-s)*np.log(1-prob), axis=1)
            s_ext = np.append(s, np.ones((s.shape[0],1)), axis=1)
            prob = sigmoid(np.dot(G_weights, s_ext.T).T)

        # Coding cost for input data.
        d_tiled = np.tile(d.astype(float), (n,1))
        costs += np.sum(
            -d_tiled*np.log(prob) - (1-d_tiled)*np.log(1-prob), axis=1)

        return costs.mean()
        
    def estimate_generative_dist(self, n = 10000):
        """ Estimate the generative distribution by sampling.
        """
        from prob import rv_bit_vector
        d = self.sample_generative_dist(n)
        return rv_bit_vector.from_samples(d)

    def sample_generative_dist(self, n, all_layers = False, top_units = None):
        """ Sample the generative distribution.

        Parameters:
        -----------
        n : int
            The number of sample to draw.

        all_layers : bool, optional [default False]
            By default, an array of input unit samples is returned. If
            'all_layers` is True, a list of sample arrays for *all* the layers,
            in top-to-bottom order, is returned instead.

        top_units : bit vector, optional
            By default, the top-level units are sampled from the generative
            biases. This parameter clamps the top-level units to specific
            values.

        Returns:
        --------
        A (list of) 2D sample array(s), where the first dimension indexes the
        individual samples. See 'all_layers' parameter.
        """
        G, G_bias = self.G, self.G_bias
        if top_units is None:
            G_bias_tiled = np.tile(G_bias, (n,1))
            d = sample_indicator(sigmoid(G_bias_tiled))
        else:
            d = np.tile(top_units, (n,1))
        if all_layers:
            samples = [ d ]
        for G_weights in G:
            d_ext = np.append(d, np.ones((d.shape[0],1)), axis=1)
            d = sample_indicator(sigmoid(np.dot(G_weights, d_ext.T).T))
            if all_layers:
                samples.append(d)
        return samples if all_layers else np.array(d, copy=0, dtype=int) 

    def sample_recognition_dist(self, d, n):
        """ Sample the recognition distribution for the given data.

        Returns:
        --------
        A list of 2D sample arrays for the hidden units, in top-to-bottom order.
        The first dimension indexes the individual samples.
        """
        R = self.R
        s = np.tile(d, (n,1))
        samples = []
        for R_weights in R:
            s_ext = np.append(s, np.ones((s.shape[0],1)), axis=1)
            s = sample_indicator(sigmoid(np.dot(R_weights, s_ext.T).T))
            samples.insert(0, s)
        return samples

    def _create_layered_network(self, topology):
        """ Create a list of weight matrices for the given network topology.
        """
        network = []
        topology = tuple(topology)
        for top, bottom in izip(topology, topology[1:]):
            network.append(np.zeros((bottom, top + 1)))
        return network

# Helmholtz machine internals
    
# Reference/fallback implementation
def _wake(world, G, G_bias, R, epsilon):
    # Sample data from the world.
    s = world()
    samples = [ s ]

    # Pass sensory data upwards through the recognition network.
    for R_weights in R:
        s = sample_indicator(sigmoid(np.dot(R_weights, np.append(s, 1))))
        samples.insert(0, s)
        
    # Pass back down through the generation network, adjusting weights as we go.
    G_bias += epsilon[0] * (samples[0] - sigmoid(G_bias))
    for G_weights, inputs, target, step \
            in izip(G, samples, samples[1:], epsilon[1:]):
        inputs = np.append(inputs, 1)
        generated = sigmoid(np.dot(G_weights, inputs))
        G_weights += step * np.outer(target - generated, inputs)

# Reference/fallback implementation
def _sleep(G, G_bias, R, epsilon):
    # Begin dreaming!
    d = sample_indicator(sigmoid(G_bias))
    dreams = [ d ]

    # Pass dream data down through the generation network.
    for G_weights in G:
        d = sample_indicator(sigmoid(np.dot(G_weights, np.append(d, 1))))
        dreams.insert(0, d)

    # Pass back up through the recognition network, adjusting weights as we go.
    for R_weights, inputs, target, step \
            in izip(R, dreams, dreams[1:], epsilon[::-1]):
        inputs = np.append(inputs, 1)
        recognized = sigmoid(np.dot(R_weights, inputs))
        R_weights += step * np.outer(target - recognized, inputs)

try:
    from _helmholtz import _wake, _sleep
except ImportError:
    print 'WARNING: Optimized HM not available.'
