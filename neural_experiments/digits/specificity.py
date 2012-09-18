# Standard library imports.
from itertools import izip

# System library imports.
import numpy as np
from traits.api import Any, Array, Dict, File, Float, Int, List, Str

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import NeuralRunner
from neural.utils.bit_vector import bit_vector_to_str
from mnist import binarize_mnist_images, read_mnist


class DigitSpecificityRunner(NeuralRunner):

    # NeuralRunner configuration.
    cls = HelmholtzMachine
    topology = [16, 128, 128, 28*28]
    rate = 0.01
    epochs = 10

    # DigitSpecificityRunner configuration.
    digits = List(Int, range(10), config=True)
    data_path = File(config=True, transient=False)

    # Results.
    dists = Dict(Str, Array(shape=(None,)))
    machine = Any(transient=True)
    specificity = Float

    # Runner interface.

    def run(self):
        # Validate settings.
        if len(self.digits) < 2:
            raise ValueError("Must use at least 2 digit classes")

        # Train the machine.
        imgs, labels = read_mnist(path=self.data_path, training=True)
        idx = np.in1d(labels, self.digits)
        imgs = binarize_mnist_images(imgs[idx])
        self.machine = machine = self.create_machine()
        self.train(machine, imgs)

        # Score the machine.
        imgs, labels = read_mnist(path=self.data_path, training=False)
        imgs = binarize_mnist_images(imgs)
        self.score(imgs, labels)

    # DigitSpecificityRunner interface.

    def score(self, imgs, labels):
        idx = np.in1d(labels, self.digits)
        imgs, labels = imgs[idx], labels[idx]

        # Compute the digit distribution for each top-level pattern.
        self.dists = dists = {}
        for img, label in izip(imgs, labels):
            top_samples = self.machine.sample_recognition_dist(img, size=10)[0]
            for sample in top_samples:
                dist = dists.setdefault(bit_vector_to_str(sample), np.zeros(10))
                dist[label] += 1
        for dist in dists.itervalues():
            dist /= dist.sum()

        # Compute the normalized digit specificity score.
        n = len(self.digits)
        scores = np.zeros(len(dists))
        for i, p in enumerate(dists.itervalues()):
            p = p[p != 0] # Set p log p = 0 when p == 0.
            scores[i] = np.sum(p * np.log(n*p))
        self.specificity = score = scores.mean() / np.log(n)
        return score


def main(args = None):
    runner = DigitSpecificityRunner()
    runner.main(args=args)
    print 'Specificity score = {}'.format(runner.specificity)

if __name__ == '__main__':
    main()
