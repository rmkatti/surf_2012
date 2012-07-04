# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import helmholtz, estimate_coding_cost
from mnist import read_mnist


def train(data_path = None):
    imgs, labels = read_mnist(path=data_path, training=True)
    imgs = imgs.astype(float) / 255
    imgs = imgs.reshape((imgs.shape[0], 28*28))

    machines = [ None ] * 10
    for digit in xrange(10):
        world = shuffled_iter(imgs[labels == digit])
        machines[digit] = helmholtz(world.next, topology = (16, 32, 32, 28*28),
                                    epsilon = 0.01, maxiter = 50000)
    return machines


def test(machines, data_path = None):
    imgs, labels = read_mnist(path=data_path, training=False)
    imgs /= 128
    imgs = imgs.reshape((imgs.shape[0], 28*28))

    # For testing only 0's and 1's.
    #sel = np.logical_or(labels == 0, labels == 1)
    #imgs = imgs[sel]
    #labels = labels[sel]

    costs = np.zeros(len(machines))
    errors = 0
    for img, label in izip(imgs, labels):
        for i, (G, G_bias, R) in enumerate(machines):
            costs[i] = estimate_coding_cost(G, G_bias, R, img, n=10)
        errors += label != np.argmin(costs)
    return float(errors) / len(imgs)


def shuffled_iter(items):
    items = np.array(items)
    while True:
        np.random.shuffle(items)
        for item in items:
            yield item


if __name__ == '__main__':
    machines = train()
    error_rate = test(machines)
    print 'Error rate = {:%}'.format(error_rate)
