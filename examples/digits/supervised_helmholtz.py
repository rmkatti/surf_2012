# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import helmholtz, estimate_coding_cost
from mnist import read_mnist


def train(digits = None, data_path = None):
    digits = digits or range(10)
    imgs, labels = read_mnist(path=data_path, training=True)
    imgs = prepare_imgs(imgs)
    machines = {}
    for digit in digits:
        world = shuffled_iter(imgs[labels == digit])
        machines[digit] = helmholtz(world.next, 
                                    topology = (16, 64, 64, 28*28),
                                    epsilon = 0.01, maxiter = 50000)
    return machines

def test(machines, data_path = None):
    imgs, labels = read_mnist(path=data_path, training=False)
    idx = np.in1d(labels, machines.keys())
    imgs = prepare_imgs(imgs[idx])
    labels = labels[idx]

    costs = np.repeat(np.inf, 10)
    errors = 0
    for img, label in izip(imgs, labels):
        for i, (G, G_bias, R) in machines.iteritems():
            costs[i] = estimate_coding_cost(G, G_bias, R, img, n=10)
        errors += label != np.argmin(costs)
    return float(errors) / len(imgs)


def prepare_imgs(imgs):
    imgs = imgs.reshape((imgs.shape[0], 28*28)).astype(float)
    imgs /= 255.0
    return np.round(imgs, out=imgs)

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
