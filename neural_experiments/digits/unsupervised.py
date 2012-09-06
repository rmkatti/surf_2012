# System library imports.
import numpy as np
from traits.api import Any, File, Float, Int, List

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import EstimatorNeuralRunner
from mnist import binarize_mnist_images, read_mnist


class UnsupervisedDigitsRunner(EstimatorNeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 128, 128, 28*28]
    rate = 0.01
    epochs = 10

    # UnsupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    coding_cost = Float
    machine = Any

    # NeuralRunner interface.

    def run(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        imgs = binarize_mnist_images(imgs)
        self.fit(imgs, labels)

        imgs, labels = read_mnist(path=self.data_path, training=False)
        imgs = binarize_mnist_images(imgs)
        self.coding_cost = -self.score(imgs, labels)

    # EstimatorNeuralRunner interface.

    def fit(self, imgs, labels):
        idx = np.in1d(labels, self.digits)
        self.machine = machine = self.create_machine()
        self.train(machine, imgs[idx])
    
    def score(self, imgs, labels):
        idx = np.in1d(labels, self.digits)
        machine = self.machine
        costs = np.array([ machine.estimate_coding_cost(img, n=10) 
                           for img in imgs[idx] ])
        return -costs.mean()


def main(args = None):
    runner = UnsupervisedDigitsRunner()
    runner.main(args=args)
    print 'Mean coding cost = {}'.format(runner.coding_cost)

if __name__ == '__main__':
    main()
