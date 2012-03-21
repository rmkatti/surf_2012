# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import estimate_generative_dist, helmholtz
from neural.util import bit_vector


def shifter():
    """ A world of two rows of short vertical line segments, with the top row
    randomly offset, but otherwise identical to, the bottom row.

    This example is used in Hinton's paper 'The Helmholtz Machine' and related
    literature. Detection of the shift corresponds to extraction of depth from
    simple stereo images.
    """
    bits = 8
    def world():
        shift = 1 if np.random.sample() < 0.5 else -1
        bottom = np.array(np.random.sample(bits) < 0.2, dtype=np.int)
        top = np.roll(bottom, shift)
        image = np.vstack((top, top, bottom, bottom))
        return image.flatten()

    G, G_bias, _ = helmholtz(world, (2, 24, 4 * bits),
                             epsilon = (0.01, 0.01, 0.15),
                             maxiter = 60000)
    samples, probs = estimate_generative_dist(G, G_bias)
    idx = np.argsort(-probs)
    for i in idx[:10]:
        print samples[i].reshape(4, bits), probs[i]
    return G, G_bias


if __name__ == '__main__':
    shifter()
