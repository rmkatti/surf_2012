# System library imports.
import numpy as np
from numpy.testing import assert_equal

# Local imports.
from neural.serialize import encode, decode


def test_serialize_array():
    x = np.arange(25).reshape(5,5)
    y = decode(encode(x))
    assert_equal(x, y)
    assert_equal(x.dtype, y.dtype)
