# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import helmholtz
from mnist import read_mnist
from util import prepare_mnist_images, shuffled_iter


def learn(digits = None, data_path = None):
    digits = digits or range(10)
    imgs, labels = read_mnist(path=data_path, training=True)
    idx = np.in1d(labels, digits)
    imgs = prepare_mnist_images(imgs[idx])
    labels = labels[idx]

    world = shuffled_iter(imgs)
    return helmholtz(world.next, 
                     topology = (len(digits), 128, 128, 28*28),
                     epsilon = 0.01, maxiter = 100000)
