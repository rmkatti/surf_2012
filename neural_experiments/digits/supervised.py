# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np
from traits.api import Any, Dict, Float, File, Int, List

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import EstimatorNeuralRunner
from mnist import binarize_mnist_images, read_mnist


class SupervisedDigitsRunner(EstimatorNeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 64, 64, 28*28]
    rate = 0.1
    anneal = 1.0
    epochs = 10

    # SupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=True)

    # Results.
    error_rate = Float
    machines = Dict(Int, Any, transient=True)

    # NeuralRunner interface.

    def run(self):
        # Fit training data.
        imgs, labels = read_mnist(path=self.data_path, training=True)
        imgs = binarize_mnist_images(imgs)
        machines = self.fit(imgs, labels)

        # Compute error rate on test set.
        imgs, labels = read_mnist(path=self.data_path, training=False)
        idx = np.in1d(labels, machines.keys())
        imgs = binarize_mnist_images(imgs[idx])
        labels = labels[idx]
        predicted = self.predict(imgs)
        self.error_rate = np.sum(predicted != labels) / float(len(imgs))

    # EstimatorNeuralRunner interface.

    def fit(self, imgs, labels):
        self.machines = {}
        for digit in self.digits:
            data = imgs[labels == digit]
            self.machines[digit] = machine = self.create_machine()
            self.train(machine, data)
        return self.machines
    
    def predict(self, imgs):
        costs = np.repeat(np.inf, 10)
        labels = np.empty(imgs.shape[0], dtype=int)
        machines = self.machines
        for i, img in enumerate(imgs):
            for j, machine in machines.iteritems():
                costs[j] = machine.estimate_coding_cost(img, n=10)
            labels[i] = np.argmin(costs)
        return labels


def main(args = None):
    runner = SupervisedDigitsRunner()
    runner.main(args=args)
    print 'Error rate = {:%}'.format(runner.error_rate)

if __name__ == '__main__':
    main()
