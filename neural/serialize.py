# System library imports.
import jsonpickle
import numpy as np

# Exported functions.
encode = jsonpickle.encode
decode = jsonpickle.decode


class open_filename(object):
    """ A context manager that opens files but passes through file-like objects.
    """
    def __init__(self, filename, *args, **kwargs):
        self.is_filename = isinstance(filename, basestring)
        if self.is_filename:
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
    """

    def flatten(self, arr, data):
        #data['bytes'] = arr.tostring()
        data['data'] = arr.tolist()
        data['dtype'] = arr.dtype.str
        #data['shape'] = arr.shape
        return data

    def restore(self, data):
        #array = np.fromstring(data['bytes'], dtype=data['dtype'])
        #return array.reshape(data['shape'])
        return np.array(data['data'], dtype=data['dtype'])

jsonpickle.handlers.registry.register(np.ndarray, NDArrayHandler)
