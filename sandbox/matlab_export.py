# Standard library imports.
import argparse
import os.path

# System library imports.
import scipy.io

# Local imports.
from neural.runner.api import Runner
from neural.utils import serialize


def export_machine(filename, machine):
    mdict = dict(cls = machine.__class__.__name__,
                 topology = machine.topology,
                 G_layers = [ machine.G_top ] + machine.G,
                 R_layers = machine.R)
    scipy.io.savemat(filename, mdict, oned_as='row')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', metavar='IN',
                        help='Input file (format: JSON)')
    parser.add_argument('outfile', metavar='OUT', nargs='?',
                        help='Output file (formt: Matlab)')
    args = parser.parse_args()
    if not args.outfile:
        root, ext = os.path.splitext(args.infile)
        args.outfile = root + '.mat'

    obj = serialize.load(args.infile)
    machine = obj.machine if isinstance(obj, Runner) else obj
    export_machine(args.outfile, machine)

if __name__ == '__main__':
    import sys
    main(sys.argv)
