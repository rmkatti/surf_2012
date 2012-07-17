# System library imports.
import numpy as np
from traits.api import Array, Int

# Local imports.
from neural.prob import kl_divergence
from runner import NeuralRunner


class BenchmarkRunner(NeuralRunner):
    """ Benchmark the Helmholtz machine on a known distribution by explicitly
    calculating the KL divergence from the distribution to the HM's generative
    distribution.
    """

    # Configuration.
    samples = Int(10000, config=True)
    yield_at = Int(500, config=True)

    # Results.
    iters = Array(dtype=int)
    kl = Array(dtype=float)

    # BaseNeuralRunner interface.
    
    def run(self):
        self.machine = machine = self.cls(topology = self.topology)

        iters, kl = [], []
        def update_kl(i):
            gen_dist = machine.estimate_generative_dist(self.samples)
            iters.append(i)
            kl.append(kl_divergence(dist, gen_dist))

        dist = self.create_dist()
        world = dist.rvs_iter(size = self.maxiter)
        machine.train(world.next, epsilon=self.epsilon, anneal=self.anneal,
                      maxiter=self.maxiter, 
                      yield_at=self.yield_at, yield_call=update_kl)

        self.iters, self.kl = iters, kl

    # BenchmarkRunner interface.
        
    def create_dist(self):
        """ Create a neural.prob.rv_bit_vector distribution for benchmarking.
        """
        raise NotImplementedError