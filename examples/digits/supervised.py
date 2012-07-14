# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import HelmholtzMachine
from mnist import read_mnist
from util import prepare_mnist_images, shuffled_iter


def train(digits = None, data_path = None, klass = None):
    digits = digits or range(10)
    klass = klass or HelmholtzMachine

    imgs, labels = read_mnist(path=data_path, training=True)
    imgs = prepare_mnist_images(imgs)
    machines = {}
    for digit in digits:
        world = shuffled_iter(imgs[labels == digit])
        machine = machines[digit] = klass(topology = (16, 64, 64, 28*28))
        machine.train(world.next, epsilon = 0.01, maxiter = 50000)
    return machines

def test(machines, data_path = None):
    imgs, labels = read_mnist(path=data_path, training=False)
    idx = np.in1d(labels, machines.keys())
    imgs = prepare_mnist_images(imgs[idx])
    labels = labels[idx]

    costs = np.repeat(np.inf, 10)
    errors = 0
    for img, label in izip(imgs, labels):
        for i, machine in machines.iteritems():
            costs[i] = machine.estimate_coding_cost(img, n=10)
        errors += label != np.argmin(costs)
    return float(errors) / len(imgs)


if __name__ == '__main__':
    machines = train()
    error_rate = test(machines)
    print 'Error rate = {:%}'.format(error_rate)
