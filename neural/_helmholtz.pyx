import numpy as np
cimport numpy as np

from neural.external import tokyo
from neural.external cimport tokyo
from prob import sample_indicator
from util import sigmoid


def wake(world, G, G_bias, R, epsilon):
    # Sample data from the world.
    s = np.ones(R[0].shape[1])
    s[:-1] = world()
    samples = [ s ]

    # Pass sensory data upwards through the recognition network.
    for R_weights in R:
        out = np.ones(R_weights.shape[0] + 1)
        sample_indicator(sigmoid(tokyo.dgemv(R_weights, s)), out[:-1])
        s = out
        samples.insert(0, s)
        
    # Pass back down through the generation network, adjusting weights as we go.
    G_bias += epsilon[0] * (samples[0][:-1] - sigmoid(G_bias))
    for G_weights, inputs, target, step \
            in zip(G, samples, samples[1:], epsilon[1:]):
        generated = sigmoid(tokyo.dgemv(G_weights, inputs))
        tokyo.dger4(step, target[:-1] - generated, inputs, G_weights)


def sleep(G, G_bias, R, epsilon):
    # Begin dreaming!
    d = np.ones(G_bias.size + 1)
    sample_indicator(sigmoid(G_bias), d[:-1])
    dreams = [ d ]

    # Pass dream data down through the generation network.
    for G_weights in G:
        out = np.ones(G_weights.shape[0] + 1)
        sample_indicator(sigmoid(tokyo.dgemv(G_weights, d)), out[:-1])
        d = out
        dreams.insert(0, d)

    # Pass back up through the recognition network, adjusting weights as we go.
    for R_weights, inputs, target, step \
            in zip(R, dreams, dreams[1:], epsilon[::-1]):
        recognized = sigmoid(tokyo.dgemv(R_weights, inputs))
        tokyo.dger4(step, target[:-1] - recognized, inputs, R_weights)
