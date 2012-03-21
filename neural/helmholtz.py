# Standard library imports.
from collections import defaultdict
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from prob import sample_indicator
from util import count_bit_vectors, sigmoid


def helmholtz(world, topology, epsilon=0.1, maxiter=50000):
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
    """
    G = create_layered_network(topology)
    G_bias = np.zeros(topology[0])
    R = create_layered_network(reversed(topology))

    if np.isscalar(epsilon):
        epsilon = np.repeat(epsilon, len(topology))
    else:
        epsilon = np.array(epsilon, copy=0)

    for i in xrange(maxiter):
        wake(world, G, G_bias, R, epsilon)
        sleep(G, G_bias, R, epsilon)

    return G, G_bias, R

def create_layered_network(topology):
    """ Create a list of weight matrices for the given layered network topology.
    """
    network = []
    topology = tuple(topology)
    for top, bottom in izip(topology, topology[1:]):
        network.append(np.zeros((bottom, top + 1)))
    return network

def estimate_generative_dist(G, G_bias, samples=100000):
    """ Estimate the probability distribution for the generative model by
    sampling from the model. Returns an array of unique generated patterns and
    an array of the corresponding estimated probabilities.
    
    This function is for analysis and testing; it is not used in the Helmoltz
    machine algorithm.
    """
    G_bias_tiled = np.tile(G_bias, (samples,1))
    d = sample_indicator(sigmoid(G_bias_tiled))
    for G_weights in G:
        d_ext = np.append(d, np.ones((d.shape[0],1)), axis=1)
        d = sample_indicator(sigmoid(np.dot(G_weights, d_ext.T).T))
    d = np.array(d, copy=0, dtype=int)
    unique_d, counts = count_bit_vectors(d)
    probs = counts / float(d.shape[0])
    return unique_d, probs
    
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
