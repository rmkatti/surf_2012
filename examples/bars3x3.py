# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import estimate_generative_dist, helmholtz
from neural.prob import rv_bit_vector


def bars3x3():
    """ A world consisting of horizontal and vertical bars in a 3x3 grid.

    This example is used in Kevin Kirby's 'Tutorial on Helmholtz Machines'.
    """
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
    world = dist.rvs

    G, G_bias, _ = helmholtz(world, (1, 6, 9),
                             epsilon = (0.01, 0.01, 0.15),
                             maxiter = 60000)
    gen_dist = estimate_generative_dist(G, G_bias)
    samples, probs = gen_dist.support
    idx = np.argsort(-probs)
    for i in idx[:patterns.shape[0]+2]:
        print samples[i], probs[i]
    return G, G_bias


if __name__ == '__main__':
    bars3x3()
