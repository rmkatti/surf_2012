# System library imports.
import numpy as np
import matplotlib.pyplot as plt

# Local imports.
from neural.tests.helmholtz_test import benchmark_helmholtz
from neural.prob import rv_bit_vector


def bars3x3():
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
                 dtype=np.float)
    p /= p.sum()
    dist = rv_bit_vector(patterns, p)

    # Benchmark the HM on this distribution.
    iters, kl = benchmark_helmholtz(dist, (1, 6, 9),
                                    epsilon = (0.01, 0.01, 0.15),
                                    maxiter = 60000, yield_at = 500)
    plt.clf()
    plt.plot(iters, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence')
    plt.show()


if __name__ == '__main__':
    bars3x3()
