# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from neural._helmholtz_laddered cimport \
    _sample_laddered_layer_1d, _probs_for_laddered_layer_1d, \
    _sample_laddered_network_1d, _probs_for_laddered_network_1d


def _wake(world, G, G_lateral, G_mean, G_var, R, R_lateral, epsilon):
    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=np.double)
    samples = _sample_laddered_network_1d(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    s = samples[0]
    generated = np.zeros(s.size)
    _probs_for_laddered_layer_1d(G_lateral[0], s, generated)

    #G_lateral[0] -= epsilon[0] * (G_lateral[0] - G_mean[0]) / G_var[0]
    _update_lateral_weights(G_lateral[0], s, generated, epsilon[0])

    G_probs = _probs_for_laddered_network_1d(G, G_lateral[1:], samples)
    for layer, lateral, inputs, target, generated, step \
            in zip(G, G_lateral[1:], samples, samples[1:], G_probs, epsilon[1:]):
        tokyo.dger4(step, inputs, target - generated, layer)
        _update_lateral_weights(lateral, target, generated, step)

def _sleep(G, G_lateral, G_mean, G_var, R, R_lateral, epsilon):
    # Generate a dream.
    d = np.zeros(G_lateral[0].shape[0])
    _sample_laddered_layer_1d(G_lateral[0], d)
    dreams = _sample_laddered_network_1d(G, G_lateral[1:], d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network_1d(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in zip(R, R_lateral, dreams, dreams[1:], R_probs, epsilon[::-1]):
        tokyo.dger4(step, inputs, target - recognized, layer)
        _update_lateral_weights(lateral, target, recognized, step)
