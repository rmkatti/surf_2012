# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import helmholtz
from mnist import read_mnist


def shuffled_iter(items):
    items = np.array(items)
    while True:
        np.random.shuffle(items)
        for item in items:
            yield item


def train(data_path = None):
    imgs, labels = read_mnist(path=data_path, training=True)
    imgs = imgs.astype(np.float_) / 255
    imgs = imgs.reshape((imgs.shape[0], 28*28))

    machines = [ None ] * 10
    for digit in xrange(1):
        world = shuffled_iter(imgs[labels == digit])
        machines[digit] = helmholtz(world.next, topology = (16, 32, 32, 28*28),
                                    epsilon = 0.01, maxiter = 50000)
    return machines
