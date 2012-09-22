import importlib, sys

# XXX: Hack for bbfreeze.
import numpy
numpy.sys = sys

name, argv = sys.argv[1], sys.argv[2:]
if not name.startswith(('neural.', 'neural_experiments.')):
    name = 'neural_experiments.' + name

module = importlib.import_module(name)
module.main(argv)
