# Standard library imports.
import datetime

# System library imports.
import numpy as np
from traits.api import HasTraits, Any, BaseInstance, Dict, Either, File, \
    Float, Int, List, Str, Type

# Local imports.
from neural.utils.io import redirect_output
import traits_argparse

# Trait definitions.
Datetime = BaseInstance(datetime.datetime)
Timedelta = BaseInstance(datetime.timedelta)


class Runner(HasTraits):
    """ The base runner class.
    """
    # Configuration.
    outfile = File(config=True, transient=True, desc='filename for run output')

    # Information.
    start_time = Datetime
    end_time = Datetime
    duration = Float
    output = Str

    def main(self, args = None):
        """ Convenience 'main' method to execute the runner.
        """
        parser = traits_argparse.make_arg_parser(self)
        parser.parse_args(args)
        try:
            self.start()
        finally:
            print self.output
        if self.outfile:
            self.save()

    def start(self):
        """ Execute the runner and store some generic information.

        Typically, subclasses should override the 'run' method, not this method.
        """
        self.start_time = datetime.datetime.now()
        try:
            with redirect_output() as io:
                return self.run()
        finally:
            self.end_time = datetime.datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
            self.output = io.getvalue()

    def run(self):
        """ Abstract method. Should be implemented by subclasses.
        """
        pass

    def save(self, filename = None):
        """ Convenience method to save the Runner using neural.serialize.
        """
        from neural.utils.serialize import save
        if filename:
            self.outfile = filename
        elif not self.outfile:
            raise ValueError('No filename specified or set.')
        save(self.outfile, self)


class NeuralRunner(Runner):
    """ The base runner class for a neural network simulation.
    """
    # Configuration: machine.
    cls = Type(config=True, config_default_module='neural.api',
               desc="class of the machine, e.g. 'LadderedHelmholtzMachine' " \
                   "or 'my_package.custom_machine.CustomMachine'")
    cls_args = Dict(Str, Any, config=True,
                    desc="optional keyword arguments for the class constructor")
    topology = List(Int, config=True, desc='layer topology of network')

    # Configuration: training.
    rate = Either(Float, List(Float), config=True, default=0.01,
                  desc="learning rate")
    anneal = Either(Float, List(Float), config=True, default=0,
                    desc="parameter for anealing schedule '" \
                        "rate = rate_0 / (1 + aneal * e)', " \
                        "where e is the current epoch")
    epochs = Int(config=True, default=100,
                 desc='number of epochs (full passes through data set)')

    def create_machine(self):
        return self.cls(topology = self.topology, **self.cls_args)

    def train(self, machine, data, **kwds):
        return machine.train(data, rate = self.rate, anneal = self.anneal,
                             epochs = self.epochs, **kwds)


class SupervisedNeuralRunner(NeuralRunner):
    """ A runner class for supervised learning with neural networks.

    For convenience, this class implements the scikits-learn supervised
    'estimator' interface. This permits the use of model selection tools like
    sklearn.grid_search.
    """

    # 'Estimator' interface
    
    def fit(self, data, target):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError

    def score(self, data, target):
        """ By default, returns the classification success rate.
        """
        predicted = self.predict(data)
        return np.sum(predicted == target) / float(len(target))

    def set_params(self, **params):
        self.trait_set(**params)
        return self
