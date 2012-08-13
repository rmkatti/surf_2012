# System library imports.
import numpy as np
from traits.api import Any, File, Int, List

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import NeuralRunner
from neural_experiments.digits.mnist import binarize_mnist_images, read_mnist


class UnsupervisedDigitsRunner(NeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 128, 128, 28*28]
    rate = 0.01
    epochs = 10

    # UnsupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    machine = Any

    def run(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        idx = np.in1d(labels, self.digits)
        imgs = binarize_mnist_images(imgs[idx])
        labels = labels[idx]

        self.machine = machine = self.create_machine()
        self.train(machine, imgs)


def main(args = None):
    runner = UnsupervisedDigitsRunner()
    runner.main(args=args)

if __name__ == '__main__':
    main()
