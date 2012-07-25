# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

import numpy as np
cimport numpy as np
from neural.external import tokyo
from neural.external cimport tokyo

from neural._helmholtz_laddered cimport \
    _sample_laddered_layer_1d, _probs_for_laddered_layer_1d, \
    _sample_laddered_network_1d, _probs_for_laddered_network_1d


cdef void _update_lateral(np.ndarray[np.double_t, ndim=2] lateral,
                          np.ndarray[np.double_t, ndim=3] param,
                          np.ndarray[np.double_t, ndim=1] target,
                          np.ndarray[np.double_t, ndim=1] probs,
                          double step):
    cdef int i, j
    for i in range(lateral.shape[0]):
        lateral[i,0] += step * (lateral[i,0] - param[0,i,0]) / param[1,i,0]
        lateral[i,0] += step * (target[i] - probs[i])
        for j in range(1, min(i+1, lateral.shape[1])):
            lateral[i,j] += step * (lateral[i,j] - param[0,i,j]) / param[1,i,j]
            lateral[i,j] += step * (target[i] - probs[i]) * target[i-j]
    return


def _wake(world, G, G_param, G_lateral, G_lateral_param, R, R_lateral, epsilon):
    # Sample data from the world.
    s = np.array(world(), copy=0, dtype=np.double)
    samples = _sample_laddered_network_1d(R, R_lateral, s)
    samples.reverse()
    
    # Pass back down through the generation network and adjust weights.
    s = samples[0]
    generated = np.zeros(s.size)
    _probs_for_laddered_layer_1d(G_lateral[0], s, generated)
    _update_lateral(G_lateral[0], G_lateral_param[0], s, generated, epsilon[0])

    G_probs = _probs_for_laddered_network_1d(G, G_lateral[1:], samples)
    for (layer, layer_param, lateral, lateral_param, 
         inputs, target, generated, step) \
    in zip(G, G_param, G_lateral[1:], G_lateral_param[1:],
           samples, samples[1:], G_probs, epsilon[1:]):
        layer += step * (layer - layer_param[0]) / layer_param[1]
        tokyo.dger4(step, inputs, target - generated, layer)
        _update_lateral(lateral, lateral_param, target, generated, step)
