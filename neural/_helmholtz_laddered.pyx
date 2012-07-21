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
                               int ladder_len,
                               np.ndarray[np.double_t, ndim=1] s_in):
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=1] s

    if ladder_len == 0:
        return sample_indicator(sigmoid(s_in))

    s = s_in.copy()
    s[0] = sample_indicator_d(logistic_d(s[0]))
    for i in range(1, s.shape[0]):
        for j in range(max(0, i - ladder_len), i):
            s[i] += lateral[i-1,j] * s[j]
        s[i] = sample_indicator_d(logistic_d(s[i]))
    return s

cdef _sample_laddered_network_1d(layers, laterals, ladder_lens, s):
    samples = [ s ]
    for layer, lateral, ladder_len in zip(layers, laterals, ladder_lens):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = _sample_laddered_layer_1d(lateral, ladder_len, s_in)
        samples.append(s)
    return samples

cdef _probs_for_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral, 
                                  int ladder_len,
                                  np.ndarray[np.double_t, ndim=1] p_in, 
                                  np.ndarray[np.double_t, ndim=1] s):
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=1] p

    if ladder_len == 0:
        return sigmoid(p_in)
    
    p = p_in.copy()
    p[0] = logistic_d(p[0])
    for i in range(1, s.shape[0]):
        for j in range(max(0, i - ladder_len), i):
            p[i] += lateral[i-1,j] * s[j]
        p[i] = logistic_d(p[i])
    return p

cdef _probs_for_laddered_network_1d(layers, laterals, ladder_lens, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, ladder_len, s in \
            zip(layers, laterals, ladder_lens, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        p = _probs_for_laddered_layer_1d(lateral, ladder_len, p_in, s)
        probs.append(p)
        s_prev = s
    return probs


cdef _update_lateral_weights(np.ndarray[np.double_t, ndim=2] lateral,
                             int ladder_len,
                             np.ndarray[np.double_t, ndim=1] target,
                             np.ndarray[np.double_t, ndim=1] probs,
                             double step):
    cdef int i, j
    if ladder_len != 0:
        for i in range(1, target.shape[0]):
            for j in range(max(0, i - ladder_len), i):
                lateral[i-1,j] += step * (target[i] - probs[i]) * target[j]
    return

def _wake(world, G, G_bias, R, G_lateral, G_bias_lateral, R_lateral,
          G_ladder_len, R_ladder_len, epsilon):
    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=np.double)
    samples = _sample_laddered_network_1d(R, R_lateral, R_ladder_len, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    generated = _probs_for_laddered_layer_1d(G_bias_lateral, G_ladder_len[0],
                                             G_bias, samples[0])
    G_bias += epsilon[0] * (samples[0] - generated)
    _update_lateral_weights(G_bias_lateral, G_ladder_len[0],
                            samples[0], generated, epsilon[0])

    G_probs = _probs_for_laddered_network_1d(G, G_lateral,
                                             G_ladder_len[1:], samples)
    for layer, lateral, ladder_len, inputs, target, generated, step \
            in zip(G, G_lateral, G_ladder_len[1:], samples, samples[1:], 
                   G_probs, epsilon[1:]):
        tokyo.dger4(step, inputs, target - generated, layer[:-1])
        layer[-1] += step * (target - generated)
        _update_lateral_weights(lateral, ladder_len, target, generated, step)

def _sleep(G, G_bias, R, G_lateral, G_bias_lateral, R_lateral,
           G_ladder_len, R_ladder_len, epsilon):
    # Generate a dream.
    d = _sample_laddered_layer_1d(G_bias_lateral, G_ladder_len[0], G_bias)
    dreams = _sample_laddered_network_1d(G, G_lateral, G_ladder_len[1:], d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network_1d(R, R_lateral,
                                             R_ladder_len, dreams)
    for layer, lateral, ladder_len, inputs, target, recognized, step \
            in zip(R, R_lateral, R_ladder_len, dreams, dreams[1:], 
                   R_probs, epsilon[::-1]):
        tokyo.dger4(step, inputs, target - recognized, layer[:-1])
        layer[-1] += step * (target - recognized)
        _update_lateral_weights(lateral, ladder_len, target, recognized, step)
