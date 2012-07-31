# Standard library imports.
import datetime

# System library imports.
from traits.api import HasTraits, BaseInstance, File, Float, Str

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
