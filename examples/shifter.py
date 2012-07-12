""" A world of two rows of short vertical line segments, with the top row
randomly offset, but otherwise identical to, the bottom row.

This example is used in Hinton's paper 'The Helmholtz Machine' and related
literature. Detection of the shift corresponds to extraction of depth from
simple stereo images.
"""
# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import HelmholtzMachine

# Constants.
shifter_bits = 8


def train_shifter():
    def world():
        shift = 1 if np.random.sample() < 0.5 else -1
        bottom = np.array(np.random.sample(shifter_bits) < 0.2, dtype=int)
        top = np.roll(bottom, shift)
        image = np.vstack((top, top, bottom, bottom))
        return image.flatten()

    machine = HelmholtzMachine(topology = (2, 24, 4 * shifter_bits))
    machine.train(world, epsilon = (0.01, 0.01, 0.15), maxiter = 60000)
    return machine
    
def estimate_most_probable(machine, n=10):
    gen_dist = machine.estimate_generative_dist(n=10000)
    samples, probs = gen_dist.support
    idx = np.argsort(-probs)
    return zip(samples[idx[:n]], probs[idx[:n]])


if __name__ == '__main__':
    from neural.ui.helmholtz_vis import HelmholtzVis

    machine = train_shifter()
    #for sample, prob in estimate_most_probable(machine):
    #    print sample.reshape(4, shifter_bits), prob

    vis = HelmholtzVis(machine = machine, 
                       layer_shapes = [(1,2), (3,8), (4, shifter_bits)])
    vis.configure_traits()
