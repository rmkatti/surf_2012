# Standard library imports.
import argparse
from collections import deque
import re
from StringIO import StringIO
import sys

# System library imports.
import numpy as np
from traits.api import BaseComplex, BaseFloat, BaseInt, BaseLong, CTrait, \
    HasTraits, Callable, Dict, Type

# Core traits_argparse API.

def add_config_arguments(parser, config_obj, exclude=[]):
    """ Dynamically construct an ArgumentParser for a HasTraits object.

    The following Trait metadata is inspected:

        'config' : If true, an argument is added for the trait (unless the trait
        name is in 'exclude')

        'config_required' : Normally, config arguments are added as optional
        arguments. If set, the argument is mandatory and hence added as a
        positional argument.

        'config_default_module' : For import traits only. Optional.
    """
    parser.config_obj = config_obj

    traits = config_obj.traits(config=True)
    names = sorted(traits.iterkeys())
    for name in names:
        if name in exclude:
            continue

        trait = traits[name]
        parse = trait.config_parse
        if parse is None:
            parse = parser_registry.lookup(trait)

        if trait.config_required:
            arg_name = name
        else:
            arg_name = '--' + name.replace('_', '-')
        metavar = name.upper().replace('_', '-')
        action = parser.add_argument(arg_name, metavar=metavar, help=trait.desc,
                                     action=_TraitsConfigAction)
        action.parse, action.trait = parse, trait

    return parser
        
class _TraitsConfigAction(argparse.Action):

    def __call__(self, parser, namespace, value, option_string=None):
        try:
            value = self.parse(self.trait, value)
            setattr(parser.config_obj, self.dest, value)
        except Exception as exc:
            raise argparse.ArgumentError(self, str(exc))

# Utility functions.

def get_trait_type(trait):
    """ Returns the Python-level TraitType for the given trait.
    """
    if isinstance(trait, CTrait):
        return trait.trait_type
    return trait

def is_number_trait(trait):
    """ Returns whether the given trait is a number type.
    """
    return isinstance(get_trait_type(trait), 
                      (BaseComplex, BaseFloat, BaseInt, BaseLong))

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

def parse_compound(trait, value):
    for handler in get_trait_type(trait).handlers:
        parse = parser_registry.lookup(handler)
        try:
            return parse(handler, value)
        except:
            pass
    # FIXME: better error reporting here.
    raise ValueError('Cannot parse string %r for compound trait.' % value)

def parse_dict(trait, value):
    typ = get_trait_type(trait)
    key_trait, val_trait = typ.key_trait, typ.value_trait
    parse_key = parser_registry.lookup(key_trait)
    parse_val = parser_registry.lookup(val_trait)

    items = {}
    for pair in _parse_sequence(value):
        pair_split = re.split('[:=]', pair)
        if len(pair_split) != 2:
            raise ValueError('Cannot parse dict pair %r.' % pair)
        key, val = pair_split
        items[parse_key(key_trait, key)] = parse_val(val_trait, val)
    return items

def parse_enum(trait, value):
    values = get_trait_type(trait).values
    candidates = filter(lambda x: x.startswith(value), values)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError('%r must identify one of %r' % (value, values))

def parse_import(trait, value):
    split = value.rsplit('.', 1)
    if len(split) == 2:
        module, name = split
    elif trait.config_default_module:
        module, name = trait.config_default_module, split[0]
    else:
        raise ValueError('No module in import string %r.' % value)
    __import__(module)
    return getattr(sys.modules[module], name)

def parse_list(trait, value):
    subtrait = get_trait_type(trait).item_trait
    parse_item = parser_registry.lookup(subtrait)
    return [ parse_item(subtrait, item) for item in _parse_sequence(value) ]

def _parse_sequence(value):
    delimiters = {']':'[', ')':'(', '}':'{'}
    current = StringIO()
    items = []

    def flush():
        items.append(current.getvalue().strip())
        current.truncate(0)

    def error():
        raise SyntaxError("Mismatched brackets in value %r" % value)

    def reset():
        del items[:]
        current.truncate(0)

    # Split on commas at zero bracket level.
    def split(value):
        stack = deque()
        for c in value.strip():
            if c in delimiters.iterkeys():
                try:
                    assert stack.pop() == delimiters[c]
                except (AssertionError, IndexError):
                    error()
            elif c in delimiters.itervalues():
                stack.append(c)
            if c == ',' and len(stack) == 0:
                flush()
            else:
                current.write(c)
        flush() if len(stack) == 0 else error()

    # For command-line convenience, we consider outer brackets optional.
    # Technically, this makes the grammar ambiguous, but we cover the usual
    # cases by backtracking.
    value = value.strip()
    start = delimiters.get(value[-1])
    if start and value[0] == start:
        try:
            split(value[1:-1])
        except SyntaxError:
            reset()
            split(value)
    else:
        split(value)

    return items

# The string->value parser registry.

class ParserRegistry(HasTraits):

    _map = Dict(Type, Callable)

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

from traits.api import Any, Array, BaseBool, BaseFile, BaseFloat, BaseInt, \
    BaseLong, BaseStr, BaseUnicode, Dict, Enum, List, TraitCompound, Type

parser_registry = ParserRegistry()
parser_registry.add(Any, make_parse(eval))
parser_registry.add(Array, parse_array)
parser_registry.add(BaseBool, make_parse(bool))
parser_registry.add(BaseFile, null_parse)
parser_registry.add(BaseFloat, make_parse(float))
parser_registry.add(BaseInt, make_parse(int))
parser_registry.add(BaseLong, make_parse(long))
parser_registry.add(BaseStr, null_parse)
parser_registry.add(BaseUnicode, null_parse)
parser_registry.add(Dict, parse_dict)
parser_registry.add(Enum, parse_enum)
parser_registry.add(List, parse_list)
parser_registry.add(TraitCompound, parse_compound)
parser_registry.add(Type, parse_import)
