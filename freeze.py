from bbfreeze import Freezer

includes = []
excludes = ['ets', 'etsdevtools', 'PIL', 'PyQt4', 'PySide', 'sip', 'wx']

freezer = Freezer('neural-0.1', includes=includes, excludes=excludes)
freezer.addScript('neural_experiments/digits/grid_search.py')
#freezer.addScript('neural_experiments/digits/supervised.py')
#freezer.addScript('neural_experiments/digits/unsupervised.py')
freezer()
