# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from util import sample_indicator, sigmoid
from _util cimport sample_indicator_d, logistic_d


cdef _sample_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral,
                               np.ndarray[np.double_t, ndim=1] s_in):
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] s

    if lateral is None:
        return sample_indicator(sigmoid(s_in))

    s = np.empty(s_in.shape[0])
    s[0] = sample_indicator_d(logistic_d(s_in[0]))
    for i in range(1, s.shape[0]):
        s_in[i] += s[:i].dot(lateral[i-1,:i])
        s[i] = sample_indicator_d(logistic_d(s_in[i]))
    return s

cdef _sample_laddered_network_1d(layers, laterals, s):
    samples = [ s ]
    for layer, lateral in zip(layers, laterals):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = _sample_laddered_layer_1d(lateral, s_in)
        samples.append(s)
    return samples

cdef _probs_for_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral, 
                                  np.ndarray[np.double_t, ndim=1] p_in, 
                                  np.ndarray[np.double_t, ndim=1] s):
    cdef int i
    if lateral is not None:
        for i in range(1, s.shape[0]):
            p_in[i] += s[:i].dot(lateral[i-1,:i])
    return sigmoid(p_in)

cdef _probs_for_laddered_network_1d(layers, laterals, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, s in zip(layers, laterals, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        p = _probs_for_laddered_layer_1d(lateral, p_in, s)
        probs.append(p)
        s_prev = s
    return probs


def _wake(world, G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    cdef int i, j
    cdef double step
    cdef np.ndarray[np.double_t, ndim=2] layer, lateral
    cdef np.ndarray[np.double_t, ndim=1] inputs, target, generated

    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=np.double)
    samples = _sample_laddered_network_1d(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    generated = _probs_for_laddered_layer_1d(G_bias_lateral, G_bias, samples[0])
    G_bias += epsilon[0] * (samples[0] - generated)
    G_probs = _probs_for_laddered_network_1d(G, G_lateral, samples)
    for layer, lateral, inputs, target, generated, step \
            in zip(G, G_lateral, samples, samples[1:], G_probs, epsilon[1:]):
        # Adjust layer weights.
        tokyo.dger4(step, inputs, target - generated, layer[:-1])
        layer[-1] += step * (target - generated)

        # Adjust lateral weights.
        if lateral is None:
            continue
        for i in range(1, target.shape[0]):
            for j in range(0, i):
                lateral[i-1,j] += step * (target[i] - generated[i]) * target[j]
    return


def _sleep(G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    cdef int i, j
    cdef double step
    cdef np.ndarray[np.double_t, ndim=2] layer, lateral
    cdef np.ndarray[np.double_t, ndim=1] inputs, target, recognized

    # Generate a dream.
    d = _sample_laddered_layer_1d(G_bias_lateral, G_bias)
    dreams = _sample_laddered_network_1d(G, G_lateral, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network_1d(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in zip(R, R_lateral, dreams, dreams[1:], R_probs, epsilon[::-1]):
        # Adjust layer weights.
        tokyo.dger4(step, inputs, target - recognized, layer[:-1])
        layer[-1] += step * (target - recognized)

        # Adjust lateral weights.
        if lateral is None:
            continue
        for i in range(1, target.shape[0]):
            for j in range(0, i):
                lateral[i-1,j] += step * (target[i] - recognized[i]) * target[j]
    return
