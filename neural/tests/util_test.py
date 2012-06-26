# System library imports.
import numpy as np
from numpy.testing import assert_equal

# Local imports.
from neural.util import bit_vector_from_str, bit_vector_to_str


def test_bit_vector_converson():
    def compare(v, s):
        assert_equal(bit_vector_from_str(s), v)

    compare(np.array([0, 1]), '01')
    compare(np.array([[0, 1], [1, 1]]), ['01', '11'])

def test_bit_vector_to_str():
    def compare(v, s):
        assert_equal(bit_vector_to_str(v), s)

    compare(np.array([0, 1]), '01')
    compare(np.array([[0, 1], [1, 1]]), ['01', '11'])
