# System library imports.
import numpy as np
from numpy.testing import assert_equal

# Local imports.
from neural.utils.bit_vector import bit_vector_from_str, bit_vector_to_str, \
    count_bit_vectors


def test_bit_vector_from_str():
    def compare(v, s):
        assert_equal(bit_vector_from_str(s), v)

    compare(np.array([0, 1]), '01')
    compare(np.array([[0, 1], [1, 1]]), ['01', '11'])

def test_bit_vector_to_str():
    def compare(v, s):
        assert_equal(bit_vector_to_str(v), s)

    compare(np.array([0, 1]), '01')
    compare(np.array([[0, 1], [1, 1]]), ['01', '11'])

def test_count_bit_vectors():
    v = np.array([ [0, 0, 1],
                   [0, 0, 1],
                   [1, 0, 1],
                   [0, 0, 1],
                   [1, 0, 1] ])
    unique_v, counts = count_bit_vectors(v)
    assert_equal(unique_v, np.array([ [0, 0, 1], [1, 0, 1] ]))
    assert_equal(counts, np.array([ 3, 2 ]))
