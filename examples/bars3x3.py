# System library imports.
import numpy as np
from scipy.stats import rv_discrete

# Local imports.
from neural.helmholtz import estimate_generative_dist, helmholtz


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
    dist = rv_discrete(name='bars3x3', values=(range(len(p)),p))
    world = lambda: patterns[dist.rvs()]

    G, G_bias, _ = helmholtz(world, (1, 6, 9),
                             epsilon = (0.01, 0.01, 0.15),
                             maxiter = 60000)
    samples, probs = estimate_generative_dist(G, G_bias)
    idx = np.argsort(-probs)
    for i in idx[:patterns.shape[0]+2]:
        print samples[i], probs[i]
    return G, G_bias


if __name__ == '__main__':
    bars3x3()
