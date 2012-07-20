# Standard library imports.
import datetime

# System library imports.
from traits.api import HasTraits, BaseInstance, Either, File, Float, Int, \
    List, Str, Type

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
    filename = File(config=True, transient=True)

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
        self.start()
        print self.output
        if self.filename:
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

    epsilon = Either(Float, List(Float), config=True, default=0.01)
    anneal = Either(Float, List(Float), config=True, default=0)
    maxiter = Int(config=True)
