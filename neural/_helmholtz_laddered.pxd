cimport numpy as np

cdef _sample_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral,
                               np.ndarray[np.double_t, ndim=1] s)
cdef _sample_laddered_network_1d(layers, laterals, s)
cdef _probs_for_laddered_layer_1d(np.ndarray[np.double_t, ndim=2] lateral, 
                                  np.ndarray[np.double_t, ndim=1] s,
                                  np.ndarray[np.double_t, ndim=1] p)
cdef _probs_for_laddered_network_1d(layers, laterals, samples)
