""" A world of two rows of short vertical line segments, with the top row
randomly offset, but otherwise identical to, the bottom row.

This example is used in Hinton's paper 'The Helmholtz Machine' and related
literature. Detection of the shift corresponds to extraction of depth from
simple stereo images.
"""
# System library imports.
import numpy as np
from traits.api import Any, Int

# Local imports.
from neural.api import HelmholtzMachine
from neural.runner.api import NeuralRunner


class ShifterRunner(NeuralRunner):

    # NeuralRunner configuration.
    cls = HelmholtzMachine
    epsilon = (0.01, 0.01, 0.15)
    maxiter = 60000

    # ShifterRunner configuration.
    bits = Int(8, config=True)

    # Results.
    machine = Any

    def _topology_default(self):
        return [2, 24, 4 * self.bits]

    def run(self):
        self.machine = machine = self.create_machine()
        machine.train(self.sample_world, epsilon=self.epsilon, 
                      anneal=self.anneal, maxiter=self.maxiter)

    def sample_world(self):
        shift = 1 if np.random.sample() < 0.5 else -1
        bottom = np.array(np.random.sample(self.bits) < 0.2, dtype=int)
        top = np.roll(bottom, shift)
        image = np.vstack((top, top, bottom, bottom))
        return image.flatten()

    
def estimate_most_probable(machine, n=10):
    gen_dist = machine.estimate_generative_dist(n=10000)
    samples, probs = gen_dist.support
    idx = np.argsort(-probs)
    return zip(samples[idx[:n]], probs[idx[:n]])

def main(args = None):
    from neural.ui.api import HelmholtzVis

    runner = ShifterRunner()
    runner.main(args=args)
    machine = runner.machine
    
    for sample, prob in estimate_most_probable(machine):
        print sample.reshape(4, runner.bits), prob


if __name__ == '__main__':
    main()
