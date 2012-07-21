# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np
from traits.api import Float, File, Int, List

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import NeuralRunner
from mnist import read_mnist
from util import prepare_mnist_images, shuffled_iter


class SupervisedDigitsRunner(NeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 64, 64, 28*28]
    epsilon = 0.01
    maxiter = 50000

    # SupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    error_rate = Float

    def run(self):
        machines = self.train()
        self.error_rate = self.test(machines)

    def train(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        imgs = prepare_mnist_images(imgs)
        machines = {}
        for digit in self.digits:
            world = shuffled_iter(imgs[labels == digit])
            machine = machines[digit] = self.cls(topology = self.topology)
            machine.train(world.next, 
                          epsilon = self.epsilon, anneal = self.anneal,
                          maxiter = self.maxiter)
        return machines

    def test(self, machines):
        imgs, labels = read_mnist(path=self.data_path, training=False)
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


def main(args = None):
    runner = SupervisedDigitsRunner()
    runner.main(args=args)
    print 'Error rate = {:%}'.format(runner.error_rate)

if __name__ == '__main__':
    main()
