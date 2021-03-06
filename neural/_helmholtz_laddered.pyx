# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from neural.utils.math import sample_indicator, sigmoid
from neural.utils._math cimport sample_indicator_d, logistic_d


cdef _sample_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral,
                               np.ndarray[np.double_t, ndim=1] s):
    cdef int i, j
    for i in range(s.shape[0]):
        s[i] += lateral[i,0]
        for j in range(1, min(i+1, lateral.shape[1])):
            s[i] += lateral[i,j] * s[i-j]
        s[i] = sample_indicator_d(logistic_d(s[i]))
    return s

cdef _sample_laddered_network_1d(layers, laterals, s):
    samples = [ s ]
    for layer, lateral in zip(layers, laterals):
        s = s.dot(layer)
        _sample_laddered_layer_1d(lateral, s)
        samples.append(s)
    return samples

cdef _probs_for_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral, 
                                  np.ndarray[np.double_t, ndim=1] s,
                                  np.ndarray[np.double_t, ndim=1] p): 
    cdef int i, j
    for i in range(s.shape[0]):
        p[i] += lateral[i,0]
        for j in range(1, min(i+1, lateral.shape[1])):
            p[i] += lateral[i,j] * s[i-j]
        p[i] = logistic_d(p[i])
    return p

cdef _probs_for_laddered_network_1d(layers, laterals, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, s in zip(layers, laterals, samples[1:]):
        p = s_prev.dot(layer)
        _probs_for_laddered_layer_1d(lateral, s, p)
        probs.append(p)
        s_prev = s
    return probs


cdef void _update_lateral(np.ndarray[np.double_t, ndim=2] lateral,
                          np.ndarray[np.double_t, ndim=1] target,
                          np.ndarray[np.double_t, ndim=1] probs,
                          double step):
    cdef int i, j
    for i in range(lateral.shape[0]):
        lateral[i,0] += step * (target[i] - probs[i])
        for j in range(1, min(i+1, lateral.shape[1])):
            lateral[i,j] += step * (target[i] - probs[i]) * target[i-j]
    return

def _wake(sample, G, G_lateral, R, R_lateral, rate):
    # Sample data from the recognition network.
    s = np.array(sample, copy=0, dtype=np.double)
    samples = _sample_laddered_network_1d(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    s = samples[0]
    generated = np.zeros(s.size)
    _probs_for_laddered_layer_1d(G_lateral[0], s, generated)
    _update_lateral(G_lateral[0], s, generated, rate[0])

    G_probs = _probs_for_laddered_network_1d(G, G_lateral[1:], samples)
    for layer, lateral, inputs, target, generated, step \
            in zip(G, G_lateral[1:], samples, samples[1:], G_probs, rate[1:]):
        tokyo.dger4(step, inputs, target - generated, layer)
        _update_lateral(lateral, target, generated, step)

def _sleep(G, G_lateral, R, R_lateral, rate):
    # Generate a dream.
    d = np.zeros(G_lateral[0].shape[0])
    _sample_laddered_layer_1d(G_lateral[0], d)
    dreams = _sample_laddered_network_1d(G, G_lateral[1:], d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network_1d(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in zip(R, R_lateral, dreams, dreams[1:], R_probs, rate[::-1]):
        tokyo.dger4(step, inputs, target - recognized, layer)
        _update_lateral(lateral, target, recognized, step)
