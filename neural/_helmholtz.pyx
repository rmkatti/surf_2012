import numpy as np
cimport numpy as np

from neural.external import tokyo
from neural.external cimport tokyo
from neural.utils.math import sample_indicator, sigmoid


def _wake(sample, G, G_top, R, rate):
    # Sample data from the recognition network.
    s = sample
    samples = [ s ]
    for R_weights in R:
        s = sample_indicator(sigmoid(s.dot(R_weights[:-1]) + R_weights[-1]))
        samples.insert(0, s)

    # Pass back down through the generation network, adjusting weights as we go.
    G_top += rate[0] * (samples[0] - sigmoid(G_top))
    for G_weights, inputs, target, step \
            in zip(G, samples, samples[1:], rate[1:]):
        generated = sigmoid(inputs.dot(G_weights[:-1]) + G_weights[-1])
        tokyo.dger4(step, inputs, target - generated, G_weights[:-1])
        G_weights[-1] += step * (target - generated)


def _sleep(G, G_top, R, rate):
    # Generate a dream.
    d = sample_indicator(sigmoid(G_top))
    dreams = [ d ]
    for G_weights in G:
        d = sample_indicator(sigmoid(d.dot(G_weights[:-1]) + G_weights[-1]))
        dreams.insert(0, d)

    # Pass back up through the recognition network, adjusting weights as we go.
    for R_weights, inputs, target, step \
            in zip(R, dreams, dreams[1:], rate[::-1]):
        recognized = sigmoid(inputs.dot(R_weights[:-1]) + R_weights[-1])
        tokyo.dger4(step, inputs, target - recognized, R_weights[:-1])
        R_weights[-1] += step * (target - recognized)
