# System library imports.
import numpy as np
import matplotlib.pyplot as plt

# Local imports.
from neural.helmholtz import HelmholtzMachine
from neural.tests.helmholtz_test import benchmark_helmholtz
from neural.prob import rv_bit_vector


def bars3x3(klass = None):
    """ A world consisting of horizontal and vertical bars in a 3x3 grid.

    This example is used in Kevin Kirby's 'Tutorial on Helmholtz Machines'.
    """
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

    # Benchmark the HM on this distribution.
    klass = klass or HelmholtzMachine
    machine = klass(topology = (1, 6, 9))
    iters, kl = benchmark_helmholtz(dist, machine,
                                    epsilon = 0.1, anneal = 1e-4,
                                    maxiter = 80000, yield_at = 500)
    print 'Final KL divergence:', kl[-1]
    plt.clf()
    plt.plot(iters, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--laddered', action='store_true', default=False,
                        help='use the laddered Helmholtz machine')
    args = parser.parse_args()
    klass = None
    if args.laddered:
        from neural.helmholtz_laddered import LadderedHelmholtzMachine
        klass = LadderedHelmholtzMachine

    bars3x3(klass = klass)
