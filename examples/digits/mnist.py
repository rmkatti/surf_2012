""" Load the MNIST handwritten digit data set as NumPy arrays. The data set is
available from http://yann.lecun.com/exdb/mnist/.

Adapted from
http://code.google.com/p/maaap-reduce/source/browse/trunk/read_mnist.py.
"""
# Standard library imports.
import os
import struct

# System library imports.
import numpy as np


def read_mnist(path = None, training = True):
    """ Load an MNIST image set and the corresponding labels.
    """
    if path is None:
        local_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(local_dir, 'data')

    prefix = 'train' if training else 't10k'
    img_path = os.path.join(path, prefix + '-images.idx3-ubyte')
    label_path = os.path.join(path, prefix + '-labels.idx1-ubyte')

    return read_mnist_images(img_path), read_mnist_labels(label_path)


def read_mnist_images(path):
    """ Load an MNIST image file.
    """
    with open(path, 'rb') as f:
        magic, num_imgs, num_rows, num_cols = struct.unpack('>iiii', f.read(16))
        assert magic == 2051, 'MNIST checksum failed'
        shape = (num_imgs, num_cols, num_rows)
        return np.fromfile(file=f, dtype=np.uint8).reshape(shape)


def read_mnist_labels(path):
    """ Load an MNIST labels file.
    """
    with open(path, 'rb') as f:
        magic, num_imgs = struct.unpack('>ii', f.read(8))
        assert magic == 2049, 'MNIST checksum failed'
        return np.fromfile(file=f, dtype=np.uint8)
