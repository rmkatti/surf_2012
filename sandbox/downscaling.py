def _wake(self, sample, data_size, iteration, rate):
    """ Run a wake cycle.
    """
    def downscale(x):
        x *= (1.0 - rate[0] * 2.5)
        x[np.abs(x) < 0.25] = 0
        
    if iteration % 5000 == 0:
        downscale(self.G_top)
        map(downscale, self.G)
        map(downscale, self.R)

    _wake(...)
