# Local imports.
from neural.helmholtz import helmholtz, estimate_generative_dist
from neural.prob import kl_divergence


def benchmark_helmholtz(world, topology, epsilon=0.1, maxiter=50000, 
                        samples=10000, yield_at=1000):
    """
    """
    p = None
    kl = []
    def update_kl(G, G_bias, R):
        d, p_G = estimate_generative_dist(G, G_bias, samples)
        kl.append(kl_divergence(p, p_G))

    helmholtz(world, topology, epsilon=epsilon, maxiter=maxiter,
              yield_at=yield_at, yield_call=update_kl)

    return np.array(kl)
