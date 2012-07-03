import numpy as np
cimport numpy as np

from neural.external import tokyo
from neural.external cimport tokyo
from prob import sample_indicator
from util import sigmoid


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
            in zip(G, samples, samples[1:], epsilon[1:]):
        inputs = np.append(inputs, 1)
        generated = sigmoid(np.dot(G_weights, inputs))
        tokyo.dger4(step, target - generated, inputs, G_weights)


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
            in zip(R, dreams, dreams[1:], epsilon[::-1]):
        inputs = np.append(inputs, 1)
        recognized = sigmoid(np.dot(R_weights, inputs))
        tokyo.dger4(step, target - recognized, inputs, R_weights)
