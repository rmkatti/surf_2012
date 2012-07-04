# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from prob import rv_bit_vector, sample_indicator
from util import sigmoid

# Helmoltz machine public API

def helmholtz(world, topology, epsilon=None, maxiter=None, 
              yield_at=None, yield_call=None):
    """ Run a Helmoltz machine.

    Parameters:
    -----------
    world : callable() -> bit vector
        A function that samples data from the "world".
        
    topology : sequence of ints
        The topology of the network. Gives the node count for each layer in
        "top-to-bottom" order. That is, the head of the sequence specifies the
        generative bias nodes, while the last element of the sequence specifies
        the input nodes. 

        Note: the number of input nodes must coincide with the size of the bit
        vectors produced by the 'world' function.
        
    epsilon : float or sequence of floats, optional [default 0.01]
        The step size for the weight update rules. If a sequence is given, it
        must be of the same size as 'topology'; a different step size may then
        be used at each layer.

    maxiter : int, optional [default 50000]
        The number the wake-sleep cycles to run.

    yield_at: int, optional [default 1000]
    yield_call : callable(iter, G, G_bias, R), optional
        If provided, the given function will be called periodically with the
        current values of the generative and recognition distributions.

    Returns:
    --------
    The generative and recognition distributions (G, G_bias, R).
    """
    epsilon = epsilon or 0.01
    if np.isscalar(epsilon):
        epsilon = np.repeat(epsilon, len(topology))
    else:
        epsilon = np.array(epsilon, copy=0)
    maxiter = maxiter or 50000
    yield_at = yield_at or 1000

    G = create_layered_network(topology)
    G_bias = np.zeros(topology[0])
    R = create_layered_network(reversed(topology))

    if yield_call:
        next_yield = yield_at
        yield_call(0, G, G_bias, R)

    for i in xrange(1, maxiter+1):
        wake(world, G, G_bias, R, epsilon)
        sleep(G, G_bias, R, epsilon)
        if yield_call and next_yield == i:
            next_yield += yield_at
            yield_call(i, G, G_bias, R)

    return G, G_bias, R

def estimate_coding_cost(G, G_bias, R, d, n = 10):
    """ Estimate the expected coding cost of the data d by sampling the
    recognition distribution.
    """
    costs = np.zeros(n)
    samples = sample_recognition_dist(R, d, n)
    
    # Coding cost for hidden units.
    prob = sigmoid(G_bias)
    for G_weights, s in izip(G, samples):
        costs += np.sum(-s*np.log(prob) - (1-s)*np.log(1-prob), axis=-1)
        s_ext = np.append(s, np.ones((s.shape[0],1)), axis=1)
        prob = sigmoid(np.dot(G_weights, s_ext.T).T)

    # Coding cost for input data.
    d_tiled = np.tile(d.astype(float), (n,1))
    costs += np.sum(-d_tiled*np.log(prob) - (1-d_tiled)*np.log(1-prob), axis=-1)

    return costs.mean()
        
def estimate_generative_dist(G, G_bias, n = 10000):
    """ Estimate the generative distribution by sampling.
    """
    d = sample_generative_dist(G, G_bias, n)
    return rv_bit_vector.from_samples(d)

def sample_generative_dist(G, G_bias, n):
    """ Sample the generative distribution.
    """
    G_bias_tiled = np.tile(G_bias, (n,1))
    d = sample_indicator(sigmoid(G_bias_tiled))
    for G_weights in G:
        d_ext = np.append(d, np.ones((d.shape[0],1)), axis=1)
        d = sample_indicator(sigmoid(np.dot(G_weights, d_ext.T).T))
    return np.array(d, copy=0, dtype=int)

def sample_recognition_dist(R, d, n):
    """ Sample the recognition distribution for the given data.

    Returns a list of sample arrays for the hidden units in top-to-bottom order.
    """
    s = np.tile(d, (n,1))
    samples = []
    for R_weights in R:
        s_ext = np.append(s, np.ones((s.shape[0],1)), axis=1)
        s = sample_indicator(sigmoid(np.dot(R_weights, s_ext.T).T))
        samples.insert(0, s)
    return samples

# Helmholz machine internals

def create_layered_network(topology):
    # Create a list of weight matrices for the given layered network topology.
    network = []
    topology = tuple(topology)
    for top, bottom in izip(topology, topology[1:]):
        network.append(np.zeros((bottom, top + 1)))
    return network
    
# Reference/fallback implementation
def wake(world, G, G_bias, R, epsilon):
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
def sleep(G, G_bias, R, epsilon):
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
    from _helmholtz import wake, sleep
except ImportError:
    print 'WARNING: Optimized HM not available.'
