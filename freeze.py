import os.path

from bbfreeze.freezer import Freezer, SharedLibrary
from bbfreeze.getdeps import getDependencies
import bbfreeze.recipes

includes = []
excludes = ['ets', 'etsdevtools', 'PIL', 'PyQt4', 'PySide', 'sip', 'wx']

# Include MKL shared libraries that cannot be detected automatically.
try:
    import mkl
except ImportError:
    pass
else:
    def add_shared_library(mf, m, filename):
        n = mf.createNode(SharedLibrary, os.path.basename(filename))
        n.filename = filename
        mf.createReference(m, n)

    def recipe_mkl(mf):
        m = mf.findNode('mkl')
        deps = getDependencies(m.filename)
        mkl_libs = filter(lambda lib: 'libmkl' in lib, deps)
        mkl_dir = os.path.split(mkl_libs[0])[0]
        for lib in ['libmkl_def.so', 'libmkl_mc.so']:
            add_shared_library(mf, m, os.path.join(mkl_dir, lib))
        
    includes.append('mkl')
    bbfreeze.recipes.recipe_mkl = recipe_mkl

freezer = Freezer('neural-0.1', includes=includes, excludes=excludes)
freezer.addScript('neural_experiments/digits/grid_search.py')
#freezer.addScript('neural_experiments/digits/supervised.py')
#freezer.addScript('neural_experiments/digits/unsupervised.py')
freezer()
