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
    pattern_dist = Dict(Str, Float)
    digit_dists = Dict(Str, Array(shape=(None,)))
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

        # Compute the digit counts for each top-level pattern.
        machine = self.machine
        sample_count = 10
        self.digit_dists = digit_dists = {}
        for img, label in izip(imgs, labels):
            samples = machine.sample_recognition_dist(img, size=sample_count)
            for top_sample in samples[0]:
                pattern = bit_vector_to_str(top_sample)
                dist = digit_dists.setdefault(pattern, np.zeros(10))
                dist[label] += 1

        # Tranform digit counts to distributions.
        total_count = len(imgs) * sample_count
        self.pattern_dist = pattern_dist = {}
        for pattern, digit_dist in digit_dists.iteritems():
            pattern_count = digit_dist.sum()
            pattern_dist[pattern] = pattern_count / total_count
            digit_dist /= pattern_count

        # Compute the normalized digit specificity score.
        score = 0.0
        n = len(self.digits)
        for pattern, weight in pattern_dist.iteritems():
            p = digit_dists[pattern]
            p = p[p != 0] # Set p log p = 0 when p == 0.
            score += weight * np.sum(p * np.log(n*p))
        score /= np.log(n)
        self.specificity = score
        return score


def main(args = None):
    runner = DigitSpecificityRunner()
    runner.main(args=args)
    print 'Specificity score = {}'.format(runner.specificity)

if __name__ == '__main__':
    main()
