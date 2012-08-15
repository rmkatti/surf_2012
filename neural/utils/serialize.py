# Standard library imports.
import base64
import os.path
import zlib

# System library imports.
import jsonpickle
import numpy as np

# Local imports.
from decorators import memoize
from io import open_filename

# Public serialization API.

def encode(value, unpicklable=True):
    """ Return a JSON formatted representation of value, a Python object.
    """
    j = Pickler(unpicklable=unpicklable)
    return jsonpickle.json.encode(j.flatten(value))

def decode(string):
    """ Convert a JSON string into a Python object.
    """
    j = Unpickler()
    return j.restore(jsonpickle.json.decode(string))

def load(filename):
    """ Load a JSON-pickled object.
    """
    with open_filename(filename, 'r') as fh:
        return decode(fh.read())

def save(filename, obj):
    """ Save an object using JSON-pickle.
    """
    with open_filename(filename, 'w') as fh:
        return fh.write(encode(obj))

class Pickler(jsonpickle.Pickler):
    """ A Pickler with fixes for NumPy and Traits.
    """
    def flatten(self, obj):
        # Handle numpy scalar types.
        if isinstance(obj, np.generic):
            return np.asscalar(obj)

        # Handle dict sub-classes (e.g. TraitsDict).
        if jsonpickle.util.is_dictionary(obj):
            self._push()
            return self._pop(self._flatten_dict_obj(obj, dict()))

        return super(Pickler, self).flatten(obj)

Unpickler = jsonpickle.Unpickler

# Monkey-patch jsonpickle. The existing implementations of these methods check
# whether the obj is strictly of the specified type, e.g. whether ``type(obj) is
# dict``. Naturally, this breaks TraitsList, TraitsDict, etc.
jsonpickle.util.is_dictionary = lambda obj: isinstance(obj, dict)
jsonpickle.util.is_list = lambda obj: isinstance(obj, list)
jsonpickle.util.is_set = lambda obj: isinstance(obj, set)

# Monkey-patch jsonpickle to infer real module name from '__main__' if possible.

def _getclassdetail(obj):
    cls = obj.__class__
    module = cls.__module__
    if module == '__main__':
        module = _get_main_name()
    name = cls.__name__
    return module, name

@memoize
def _get_main_name():
    import __main__
    module = '__main__'
    try:
        # Available if called with 'python -m'.
        module = __main__.__loader__.fullname
    except AttributeError:
        try:
            # Available under most circumstances.
            path = os.path.abspath(__main__.__file__)
        except AttributeError:
            pass
        else:
            path, name = os.path.split(path)
            names = [ os.path.splitext(name)[0] ]
            while path and os.path.isfile(os.path.join(path, '__init__.py')):
                path, name = os.path.split(path)
                names.insert(0, name)
            module = '.'.join(names)
    return module

jsonpickle.pickler._getclassdetail = _getclassdetail

# Add np.ndarray support to jsonpickle. Not a monkey-patch!

class NDArrayHandler(jsonpickle.handlers.BaseHandler):
    """ A JSON-pickler for NumPy arrays.

    The raw bytes are compressed using zlib and then base 64 encoded.
    """

    def flatten(self, arr, data):
        data['bytes'] = base64.b64encode(zlib.compress(arr.tostring()))
        data['dtype'] = arr.dtype.str
        data['shape'] = arr.shape
        return data

    def restore(self, data):
        byte_str = zlib.decompress(base64.b64decode(data['bytes']))
        array = np.fromstring(byte_str, dtype=data['dtype'])
        return array.reshape(data['shape'])

jsonpickle.handlers.registry.register(np.ndarray, NDArrayHandler)
