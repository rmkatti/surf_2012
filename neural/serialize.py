# Standard library imports.
import base64
import os.path
import zlib

# System library imports.
import jsonpickle
import numpy as np

# Exported functions.
encode = jsonpickle.encode
decode = jsonpickle.decode

# Monkey-patch jsonpickle. The existing implementations of these methods check
# whether the obj is strictly of the specified type, e.g. whether ``type(obj) is
# dict``. Naturally, this breaks TraitsList, TraitsDict, etc.
jsonpickle.util.is_dictionary = lambda obj: isinstance(obj, dict)
jsonpickle.util.is_list = lambda obj: isinstance(obj, list)
jsonpickle.util.is_set = lambda obj: isinstance(obj, set)


class open_filename(object):
    """ A context manager that opens files but passes through file-like objects.
    """
    def __init__(self, filename, *args, **kwargs):
        self.is_filename = isinstance(filename, basestring)
        if self.is_filename:
            filename = os.path.abspath(os.path.expanduser(filename))
            self.fh = open(filename, *args, **kwargs)
        else:
            self.fh = filename

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_filename:
            self.fh.close()
        return False


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
