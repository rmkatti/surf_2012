# System library imports.
import numpy as np

# Local imports.
from neural.helmholtz import helmholtz, estimate_generative_dist
from neural.prob import kl_divergence


def benchmark_helmholtz(dist, topology, epsilon=None, maxiter=None, 
                        samples=None, yield_at=None):
    """ Benchmark the Helmholtz machine on a known distribution.
    """
    iters, kl = [], []
    samples = samples or 10000
    def update_kl(i, G, G_bias, R):
        gen_dist = estimate_generative_dist(G, G_bias, samples)
        iters.append(i)
        kl.append(kl_divergence(dist, gen_dist))

    world = dist.rvs
    helmholtz(world, topology, epsilon=epsilon, maxiter=maxiter,
              yield_at=yield_at, yield_call=update_kl)

    return np.array(iters), np.array(kl)
