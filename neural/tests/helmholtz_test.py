# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import HelmholtzMachine
from neural.prob import kl_divergence


def benchmark_helmholtz(dist, machine, epsilon=None, maxiter=None, 
                        samples=None, yield_at=None):
    """ Benchmark the Helmholtz machine on a known distribution.
    """
    iters, kl = [], []
    samples = samples or 10000
    def update_kl(i):
        gen_dist = machine.estimate_generative_dist(samples)
        iters.append(i)
        kl.append(kl_divergence(dist, gen_dist))

    world = dist.rvs
    machine.train(world, epsilon=epsilon, maxiter=maxiter,
                  yield_at=yield_at, yield_call=update_kl)

    return np.array(iters), np.array(kl)
