# Standard library imports.
from collections import defaultdict
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from prob import rv_bit_vector, sample_indicator
from util import sigmoid


def helmholtz(world, topology, epsilon=0.1, maxiter=50000, 
              yield_at=1000, yield_call=None):
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
        
    epsilon : float or sequence of floats, optional
        The step size for the weight update rules. If a sequence is given, it
        must be of the same size as 'topology'; a different step size may then
        be used at each layer.

    maxiter : int, optional
        The number the wake-sleep cycles to run.

    yield_at: int, optional
    yield_call : callable(G, G_bias, R), optional
        If provided, the given function will be called periodically with the
        current values of the generative and recognition distributions.

    Returns:
    --------
    The generative and recognition distributions (G, G_bias, R).
    """
    G = create_layered_network(topology)
    G_bias = np.zeros(topology[0])
    R = create_layered_network(reversed(topology))

    if np.isscalar(epsilon):
        epsilon = np.repeat(epsilon, len(topology))
    else:
        epsilon = np.array(epsilon, copy=0)

    next_yield = yield_at - 1 if yield_call else -1
    for i in xrange(maxiter):
        wake(world, G, G_bias, R, epsilon)
        sleep(G, G_bias, R, epsilon)
        if next_yield == i:
            next_yield += yield_at
            yield_call(G, G_bias, R)

    return G, G_bias, R

def sample_generative_dist(G, G_bias, n):
    """ Sample the generative distribution.
    """
    G_bias_tiled = np.tile(G_bias, (n,1))
    d = sample_indicator(sigmoid(G_bias_tiled))
    for G_weights in G:
        d_ext = np.append(d, np.ones((d.shape[0],1)), axis=1)
        d = sample_indicator(sigmoid(np.dot(G_weights, d_ext.T).T))
    return np.array(d, copy=0, dtype=int)

def estimate_generative_dist(G, G_bias, samples=10000):
    """ Estimate the generative distribution by sampling.
    """
    d = sample_generative_dist(G, G_bias, samples)
    return rv_bit_vector.from_samples(d)


def create_layered_network(topology):
    # Create a list of weight matrices for the given layered network topology.
    network = []
    topology = tuple(topology)
    for top, bottom in izip(topology, topology[1:]):
        network.append(np.zeros((bottom, top + 1)))
    return network
    
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
