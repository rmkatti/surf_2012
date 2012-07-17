# Standard library imports.
import datetime

# System library imports.
from traits.api import HasTraits, BaseInstance, Either, File, Float, Int, \
    List, Type

# Local imports.
import traits_argparse

# Trait definitions.
Datetime = BaseInstance(datetime.datetime)
Timedelta = BaseInstance(datetime.timedelta)


class Runner(HasTraits):
    """ The base runner class.
    """
    # Configuration.
    filename = File(config=True, transient=True)

    # Information.
    start_time = Datetime
    end_time = Datetime
    duration = Float

    def main(self, args=None):
        """ Convenience 'main' method to execute to runner.
        """
        parser = traits_argparse.make_arg_parser(self)
        parser.parse_args(args)
        self.start()
        if self.filename:
            self.save()

    def start(self):
        """ Execute the runner and store some generic statistics.

        Typically, subclasses should override the 'run' method, not this method.
        """
        self.start_time = datetime.datetime.now()
        try:
            return self.run()
        finally:
            self.end_time = datetime.datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()

    def run(self):
        """ Abstract method. Should be implemented by subclasses.
        """
        pass

    def save(self, filename = None):
        """ Convenience method to save the Runner using neural.serialize.
        """
        from neural.serialize import save
        if filename:
            self.filename = filename
        elif not self.filename:
            raise ValueError('No filename specified or set.')
        save(self.filename, self)


class NeuralRunner(Runner):
    """ The base runner class for a neural network simulation.
    """
    # Configuration.
    cls = Type(config=True)
    topology = List(Int, config=True)

    epsilon = Either(Float, List(Float), config=True)
    anneal = Either(Float, List(Float), config=True)
    maxiter = Int(config=True)
