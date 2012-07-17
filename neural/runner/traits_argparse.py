# Standard library imports.
import argparse
import sys

# System library imports.
import numpy as np
from traits.api import CTrait, HasTraits, Class, Callable, Dict

# Dynamic ArgumentParser construction.

def make_arg_parser(config_obj, *args, **kwds):
    """ Dynamically construct an ArgumentParser for a HasTraits object.
    """
    parser = argparse.ArgumentParser(*args, **kwds)
    parser.config_obj = config_obj

    for name, trait in config_obj.traits(config=True).iteritems():
        parse = trait.config_parse
        if parse is None:
            parse = parser_registry.lookup(trait)
        def convert(val, parse=parse, trait=trait):
            return parse(trait, val)
        parser.add_argument('--' + name, help=trait.desc, type=convert,
                            action=RunConfigAction)

    return parser
        
class RunConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(parser.config_obj, self.dest, values)

# Built-in string->value parsers.

def make_parse(typ):
    return lambda trait, value: typ(value)

def null_parse(trait, value):
    return value

def parse_array(trait, value):
    arr = np.array(_parse_sequence(value), dtype=trait.dtype)
    if trait.shape:
        arr = arr.reshape(trait.shape)
    return arr

def parse_class(trait, value):
    module, name = value.rsplit('.', 1)
    __import__(module)
    return getattr(sys.modules[module], name)

def parse_compound(trait, value):
    for handler in get_trait_type(trait).handlers:
        parse = parser_registry.lookup(handler)
        try:
            return parse(handler, value)
        except:
            pass
    # FIXME: better error reporting here.
    raise ValueError('Cannot parse string %r for compound trait.' % value)

def parse_list(trait, value):
    subtrait = get_trait_type(trait).item_trait
    parse_item = parser_registry.lookup(subtrait)
    return [ parse_item(subtrait, item) for item in _parse_sequence(value) ]

def _parse_sequence(value):
    return [ item.strip() for item in value.strip('[]()').split(',') ]

# The string->value parser registry.

def get_trait_type(trait):
    if isinstance(trait, CTrait):
        return trait.trait_type
    return trait

class ParserRegistry(HasTraits):

    _map = Dict(Class, Callable)

    def add(self, cls, parse):
        self._map[cls] = parse

    def lookup(self, trait):
        typ = type(get_trait_type(trait))
        for cls in typ.mro():
            if cls in self._map:
                return self._map[cls]
        raise NotImplementedError('No parser for trait type %r' % typ.__name__)

    def remove(self, cls):
        if cls in self._map:
            del self._map[cls]

from traits.api import Array, BaseBool, BaseFile, BaseFloat, BaseInt, \
    BaseLong, BaseStr, BaseUnicode, Class, List, TraitCompound

parser_registry = ParserRegistry()
parser_registry.add(Array, parse_array)
parser_registry.add(BaseBool, make_parse(bool))
parser_registry.add(BaseFile, null_parse)
parser_registry.add(BaseFloat, make_parse(float))
parser_registry.add(BaseInt, make_parse(int))
parser_registry.add(BaseLong, make_parse(long))
parser_registry.add(BaseStr, null_parse)
parser_registry.add(BaseUnicode, null_parse)
parser_registry.add(Class, parse_class)
parser_registry.add(List, parse_list)
parser_registry.add(TraitCompound, parse_compound)
