# System library imports.
import numpy as np
from numpy.testing import assert_equal
from traits.api import HasTraits, Array, Either, Float, Int, List, Str, Type

# Local imports.
from neural.runner.traits_argparse import parse_array, parse_compound, \
    parse_list, parse_type


def test_parse_array():
    trait = Array(dtype=int)
    x = np.arange(5)
    for y in [ parse_array(trait, '[0,1,2,3,4]'),
               parse_array(trait, '(0, 1, 2, 3, 4)'),
               parse_array(trait, '0,1,2,3,4') ]:
        assert_equal(x, y)
        assert_equal(x.dtype, y.dtype)

def test_parse_compound():
    class Object(HasTraits):
        attr = Either(Int, Str)
    obj = Object()
    trait = obj.trait('attr')

    assert_equal(0, parse_compound(trait, '0'))
    assert_equal(10, parse_compound(trait, '10'))
    assert_equal('foo', parse_compound(trait, 'foo'))

def test_parse_list():
    trait = List(Float)
    x = [1.0, 1.5, 2.0]
    for y in [ parse_list(trait, '[1.0,1.5,2.0]'),
               parse_list(trait, '(1.0, 1.5, 2.0)'),
               parse_list(trait, '1.0,1.5,2.0') ]:
        assert_equal(x, y)

def test_parse_type():
    import datetime
    trait = Type()
    assert_equal(datetime.datetime,
                 parse_type(trait, 'datetime.datetime'))
