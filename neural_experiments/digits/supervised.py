# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np
from traits.api import Float, File, Int, List

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import NeuralRunner
from mnist import binarize_mnist_images, read_mnist


class SupervisedDigitsRunner(NeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 64, 64, 28*28]
    rate = 0.01
    epochs = 10

    # SupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    error_rate = Float

    def run(self):
        machines = self.train_machines()
        self.error_rate = self.test_machines(machines)

    def train_machines(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        imgs = binarize_mnist_images(imgs)
        machines = {}
        for digit in self.digits:
            data = imgs[labels == digit]
            machine = machines[digit] = self.create_machine()
            self.train(machine, data)
        return machines

    def test_machines(self, machines):
        imgs, labels = read_mnist(path=self.data_path, training=False)
        idx = np.in1d(labels, machines.keys())
        imgs = binarize_mnist_images(imgs[idx])
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
