# Standard library imports.
import argparse
import os.path
import sys

# System library imports.
import numpy as np
from traits.api import Dict, Float, Instance, Int, List, Str, Type

# Local imports.
from runner import Runner
import traits_argparse


class TrialRunner(Runner):
    """ A meta-runner for performing multiple trials of a given runner.
    """
    # Configuration.
    runner_cls = Type(Runner, config=True, config_required=True,
                      desc="runner class to use for each trial")
    trials = Int(10, config=True, desc="number of trials")
    stats = List(Str, config=True, transient=True,
                 desc="compute statistics for these runner attributes")

    # Runtime.
    base_runner = Instance(Runner)

    # Results.
    results = List(Runner)
    mean = Dict(Str, Float, desc="sample mean")
    error = Dict(Str, Float, desc="sample standard deviation of the mean")

    # Runner interface

    def run(self):
        for i in xrange(1, self.trials+1):
            if self.verbose:
                print 'Running trial #{}'.format(i)
            runner = self.base_runner.clone_traits()
            runner.start()
            self.results.append(runner)

        for attr in self.stats:
            getter = lambda runner: getattr(runner, attr)
            values = np.array(map(getter, self.results))
            self.mean[attr] = np.mean(values)
            self.error[attr] = np.std(values) / np.sqrt(values.size)
        
        if self.verbose and self.stats:
            print 'Statistics for {} trials:'.format(self.trials)
            template = u'{} = {} \u00B1 {}'
            for attr in self.stats:
                print template.format(attr, self.mean[attr], self.error[attr])

    def parse_args(self, args = None):
        # Parse TrialRunner arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', action='count', default=0,
                            help='verbosity level')
        traits_argparse.add_config_arguments(parser, self)
        parser.add_argument('runner_args', metavar='RUNNER-ARGS',
                            nargs=argparse.REMAINDER,
                            help='arguments for runner '
                                 '(see available arguments with -h)')
        namespace = parser.parse_args(args)
        self.verbose = self.base_runner.verbose = namespace.verbose

        # Parse sub-Runner arguments.
        prog = '{} {}'.format(os.path.basename(sys.argv[0]),
                              self.base_runner.__class__.__name__)
        parser = argparse.ArgumentParser(prog=prog)
        traits_argparse.add_config_arguments(parser, self.base_runner, 
                                             exclude=['outfile'])
        parser.parse_args(namespace.runner_args)

    # TrialRunner interface

    def _base_runner_default(self):
        return self.runner_cls()


def main(args = None):
    runner = TrialRunner()
    runner.main(args=args)

if __name__ == '__main__':
    main()
