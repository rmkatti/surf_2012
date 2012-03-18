# Standard library imports.
from collections import defaultdict
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from prob import sample_indicator
from util import encode_bit_vector, sigmoid


def helmholtz(world, topology, epsilon=0.1, iterations=50000):
    """ Run a Helmoltz machine.
    """
    G = create_layered_network(topology)
    G_bias = np.zeros(topology[0])
    R = create_layered_network(reversed(topology))

    for i in xrange(iterations):
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
    sampling from the model. Returns an array of estimated probabilities,
    indexed by the integer encoding of the sample data.
    
    This function is for analysis and testing; it is not used in the Helmoltz
    machine algorithm.
    """
    G_bias_tiled = np.tile(G_bias, (samples,1))
    d = sample_indicator(sigmoid(G_bias_tiled))
    for G_weights in G:
        d_ext = np.append(d, np.ones((d.shape[0],1)), axis=1)
        d = sample_indicator(sigmoid(np.dot(G_weights, d_ext.T).T))
    d = np.array(d, dtype=int)
    counts = np.bincount(encode_bit_vector(d))
    probs = np.zeros(2 ** d.shape[1])
    probs[:counts.size] = counts / float(d.shape[0])
    return probs
    
def wake(world, G, G_bias, R, epsilon):
    # Sample data from the world.
    s = world()
    samples = [ s ]

    # Pass sensory data upwards through the recognition network.
    for R_weights in R:
        s = sample_indicator(sigmoid(np.dot(R_weights, np.append(s, 1))))
        samples.insert(0, s)
        
    # Pass back down through the generation network, adjusting weights as we go.
    G_bias += epsilon * (samples[0] - sigmoid(G_bias))
    for G_weights, inputs, target in izip(G, samples, samples[1:]):
        inputs = np.append(inputs, 1)
        generated = sigmoid(np.dot(G_weights, inputs))
        G_weights += epsilon * np.outer(target - generated, inputs)

def sleep(G, G_bias, R, epsilon):
    # Begin dreaming!
    d = sample_indicator(sigmoid(G_bias))
    dreams = [ d ]

    # Pass dream data down through the generation network.
    for G_weights in G:
        d = sample_indicator(sigmoid(np.dot(G_weights, np.append(d, 1))))
        dreams.insert(0, d)

    # Pass back up through the recognition network, adjusting weights as we go.
    for R_weights, inputs, target in izip(R, dreams, dreams[1:]):
        inputs = np.append(inputs, 1)
        recognized = sigmoid(np.dot(R_weights, inputs))
        R_weights += epsilon * np.outer(target - recognized, inputs)
