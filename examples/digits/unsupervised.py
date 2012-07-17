# System library imports.
import numpy as np
from traits.api import Any, File, Int, List

# Local imports.
from neural.helmholtz import HelmholtzMachine
from neural.runner.runner import NeuralRunner
from mnist import read_mnist
from util import prepare_mnist_images, shuffled_iter


class UnsupervisedDigitsRunner(NeuralRunner):
    
    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = (16, 128, 128, 28*28)
    epsilon = 0.01
    maxiter = 100000

    # UnsupervisedDigitsRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    machine = Any

    def run(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        idx = np.in1d(labels, self.digits)
        imgs = prepare_mnist_images(imgs[idx])
        labels = labels[idx]

        world = shuffled_iter(imgs, copy=False)
        self.machine = self.cls(topology = self.topology)
        self.machine.train(world.next, 
                           epsilon = self.epsilon, anneal = self.anneal,
                           maxiter = self.maxiter)


def main(args = None):
    runner = UnsupervisedDigitsRunner()
    runner.main(args=args)

if __name__ == '__main__':
    main()
