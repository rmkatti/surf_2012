cimport cython
import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from util import sigmoid
from _util cimport logistic_d, sample_indicator_d


cdef _sample_laddered_bias_1d(bias, lateral):
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] s, s_in

    s = np.empty(bias.shape)
    s_in = bias.copy()
    s[0] = sample_indicator_d(logistic_d(s_in[0]))
    for i in range(1, len(s)):
        s_in[i] += s[:i].dot(lateral[i-1,:i])
        s[i] = sample_indicator_d(logistic_d(s_in[i]))
    return s

cdef _sample_laddered_network_1d(layers, laterals, 
                                 np.ndarray[np.double_t, ndim=1] s):
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] s_in

    samples = [ s ]
    for layer, lateral in zip(layers, laterals):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = np.empty(s_in.size)
        s[0] = sample_indicator_d(logistic_d(s_in[0]))
        for i in range(1, len(s)):
            s_in[i] += s[:i].dot(lateral[i-1,:i])
            s[i] = sample_indicator_d(logistic_d(s_in[i]))
        samples.append(s)
    return samples

cdef _probs_for_laddered_bias_1d(bias, lateral, s):
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] p_in

    p_in = bias.copy()
    for i in range(1, len(s)):
        p_in[i] += s[:i].dot(lateral[i-1,:i])
    return sigmoid(p_in)

cdef _probs_for_laddered_network_1d(layers, laterals, samples):
    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] p, p_in, s, s_prev

    probs = []
    s_prev = samples[0]
    for layer, lateral, s in zip(layers, laterals, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        for i in range(1, len(s)):
            p_in[i] += s[:i].dot(lateral[i-1,:i])
        p = sigmoid(p_in)
        s_prev = s
        probs.append(p)
    return probs

@cython.boundscheck(False)
def _wake(world, G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    cdef int i, j
    cdef double step
    cdef np.ndarray[np.double_t, ndim=2] layer, lateral
    cdef np.ndarray[np.double_t, ndim=1] inputs, target, generated

    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=float)
    samples = _sample_laddered_network_1d(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    generated = _probs_for_laddered_bias_1d(G_bias, G_bias_lateral, samples[0])
    G_bias += epsilon[0] * (samples[0] - generated)
    G_probs = _probs_for_laddered_network_1d(G, G_lateral, samples)
    for layer, lateral, inputs, target, generated, step \
            in zip(G, G_lateral, samples, samples[1:], G_probs, epsilon[1:]):
        tokyo.dger4(step, inputs, target - generated, layer[:-1])
        layer[-1] += step * (target - generated)
        for i in range(1, len(target)):
            for j in range(0, i):
                lateral[i-1,j] += step * (target[i] - generated[i]) * target[j]
    return

@cython.boundscheck(False)
def _sleep(G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    cdef int i, j
    cdef double step
    cdef np.ndarray[np.double_t, ndim=2] layer, lateral
    cdef np.ndarray[np.double_t, ndim=1] inputs, target, recognized

    # Generate a dream.
    d = _sample_laddered_bias_1d(G_bias, G_bias_lateral)
    dreams = _sample_laddered_network_1d(G, G_lateral, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network_1d(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in zip(R, R_lateral, dreams, dreams[1:], R_probs, epsilon[::-1]):
        tokyo.dger4(step, inputs, target - recognized, layer[:-1])
        layer[-1] += step * (target - recognized)
        for i in range(1, len(target)):
            for j in range(0, i):
                lateral[i-1,j] += step * (target[i] - recognized[i]) * target[j]
    return
