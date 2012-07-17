""" A world consisting of horizontal and vertical bars in a 3x3 grid.

This example is used in Kevin Kirby's 'Tutorial on Helmholtz Machines'.
"""
# System library imports.
import numpy as np

# Local imports.
from neural.runner.benchmark import BenchmarkRunner
from neural.helmholtz import HelmholtzMachine
from neural.prob import rv_bit_vector


class Bars3x3Runner(BenchmarkRunner):

    # BenchmarkRunner configuration.
    cls = HelmholtzMachine
    topology = (1, 6, 9)
    epsilon = 0.1
    anneal = 1e-4
    maxiter = 80000
    yield_at = 500

    def create_dist(self):
        # Construct the bars distribution.
        patterns = np.array([ # Vertical bars.
                              [0, 0, 1, 0, 0, 1, 0, 0, 1],
                              [0, 1, 1, 0, 1, 1, 0, 1, 1],
                              [1, 0, 0, 1, 0, 0, 1, 0, 0],
                              [1, 1, 0, 1, 1, 0, 1, 1, 0],
                              [0, 1, 0, 0, 1, 0, 0, 1, 0],
                              [1, 0, 1, 1, 0, 1, 1, 0, 1],
                              # Horizontal bars.
                              [0, 0, 0, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 1, 1],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0] ])
        p = np.array([ 2, 2, 2, 2, 2, 2,   # high probability
                       1, 1, 1, 1, 1, 1 ], # low probability
                     dtype=float)
        p /= p.sum()
        dist = rv_bit_vector(patterns, p)
        return dist


def main(args = None):
    import matplotlib.pyplot as plt

    runner = Bars3x3Runner()
    runner.main(args=args)    
    iters, kl = runner.iters, runner.kl

    print 'Final KL divergence:', kl[-1]
    plt.clf()
    plt.plot(iters, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.show()


if __name__ == '__main__':
    main()
