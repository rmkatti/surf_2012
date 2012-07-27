# System library imports.
import numpy as np
from traits.api import Any, Array, Int

# Local imports.
from neural.prob import kl_divergence
from runner import NeuralRunner


class BenchmarkRunner(NeuralRunner):
    """ Benchmark the Helmholtz machine on a known distribution by explicitly
    calculating the KL divergence from the distribution to the HM's generative
    distribution.
    """

    # Configuration.
    benchmark_interval = Int(500, config=True)
    data_size = Int(10000, config=True)

    # Results.
    machine = Any
    iters = Array(dtype=int)
    kl = Array(dtype=float)

    # BaseNeuralRunner interface.
    
    def run(self):
        self.machine = machine = self.create_machine()
        dist = self.create_dist()

        iters, kl = [], []
        def update_kl(i):
            gen_dist = machine.estimate_generative_dist(self.data_size)
            iters.append(i)
            kl.append(kl_divergence(dist, gen_dist))

        data = dist.rvs(size = self.data_size)
        self.train(machine, data,
                   yield_at = self.benchmark_interval, yield_call = update_kl)
        self.iters, self.kl = iters, kl

    # BenchmarkRunner interface.
        
    def create_dist(self):
        """ Create a neural.prob.rv_bit_vector distribution for benchmarking.
        """
        raise NotImplementedError
