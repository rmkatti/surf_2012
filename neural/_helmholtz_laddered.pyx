cimport cython
import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from util import sample_indicator, sigmoid


def _sample_laddered_layer(lateral, s_in):
    if lateral is None:
        return sample_indicator(sigmoid(s_in))
    s = np.empty(s_in.shape)
    s[...,0] = sample_indicator(sigmoid(s_in[...,0]))
    for i in range(1, s.shape[-1]):
        s_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
        s[...,i] = sample_indicator(sigmoid(s_in[...,i]))
    return s

def _sample_laddered_network(layers, laterals, s):
    samples = [ s ]
    for layer, lateral in zip(layers, laterals):
        s_in = s.dot(layer[:-1]) + layer[-1]
        s = _sample_laddered_layer(lateral, s_in)
        samples.append(s)
    return samples

def _probs_for_laddered_layer(lateral, p_in, s):
    if lateral is not None:
        for i in range(1, s.shape[-1]):
            p_in[...,i] += s[...,:i].dot(lateral[i-1,:i])
    return sigmoid(p_in)

def _probs_for_laddered_network(layers, laterals, samples):
    probs = []
    s_prev = samples[0]
    for layer, lateral, s in zip(layers, laterals, samples[1:]):
        p_in = s_prev.dot(layer[:-1]) + layer[-1]
        p = _probs_for_laddered_layer(lateral, p_in, s)
        probs.append(p)
        s_prev = s
    return probs


@cython.boundscheck(False)
def _wake(world, G, G_bias, R, G_lateral, G_bias_lateral, R_lateral, epsilon):
    cdef int i, j
    cdef double step
    cdef np.ndarray[np.double_t, ndim=2] layer, lateral
    cdef np.ndarray[np.double_t, ndim=1] inputs, target, generated

    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=float)
    samples = _sample_laddered_network(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    generated = _probs_for_laddered_layer(G_bias_lateral, G_bias, samples[0])
    G_bias += epsilon[0] * (samples[0] - generated)
    G_probs = _probs_for_laddered_network(G, G_lateral, samples)
    for layer, lateral, inputs, target, generated, step \
            in zip(G, G_lateral, samples, samples[1:], G_probs, epsilon[1:]):
        # Adjust layer weights.
        tokyo.dger4(step, inputs, target - generated, layer[:-1])
        layer[-1] += step * (target - generated)
        # Adjust lateral weights.
        if lateral is None:
            continue
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
    d = _sample_laddered_layer(G_bias_lateral, G_bias)
    dreams = _sample_laddered_network(G, G_lateral, d)
    dreams.reverse()

    # Pass back up through the recognition network and adjust weights.
    R_probs = _probs_for_laddered_network(R, R_lateral, dreams)
    for layer, lateral, inputs, target, recognized, step \
            in zip(R, R_lateral, dreams, dreams[1:], R_probs, epsilon[::-1]):
        # Adjust layer weights.
        tokyo.dger4(step, inputs, target - recognized, layer[:-1])
        layer[-1] += step * (target - recognized)
        # Adjust lateral weights.
        if lateral is None:
            continue
        for i in range(1, len(target)):
            for j in range(0, i):
                lateral[i-1,j] += step * (target[i] - recognized[i]) * target[j]
    return
