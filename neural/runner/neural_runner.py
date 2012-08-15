# System library imports.
import numpy as np
from traits.api import Any, Dict, Either, Float, Int, List, Str, Type

# Local imports.
from runner import Runner


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


class EstimatorNeuralRunner(NeuralRunner):
    """ A NeuralRunner that implements the scikit-learn 'estimator' interface.

    This permits the runner to be use with model selection tools like
    sklearn.grid_search.
    """

    # sklearn 'estimator' interface
    
    def get_params(self, deep=True):
        return { name : getattr(self, name, None)
                 for name in self._get_param_names() }

    def _get_param_names(self):
        names = self.trait_names()
        ignore = ['trait_added', 'trait_modified']
        return list(set(names).difference(ignore))

    def set_params(self, **params):
        self.trait_set(**params)
        return self
    
    def fit(self, data, target):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError

    def score(self, data, target):
        """ By default, returns the classification success rate.
        """
        predicted = self.predict(data)
        return np.sum(predicted == target) / float(len(target))
