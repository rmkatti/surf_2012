from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = 'surf_2012',
    version = '0.1',
    cmdclass = {'build_ext': build_ext},
    packages = find_packages(),
    ext_modules = [
        Extension('neural.external.tokyo',
                  ['neural/external/tokyo.pyx'],
                  libraries = ['blas']),
        Extension('neural._helmholtz',
                  ['neural/_helmholtz.pyx']),
        Extension('neural._helmholtz_laddered',
                  ['neural/_helmholtz_laddered.pyx']),
        Extension('neural._util',
                  ['neural/_util.pyx'],
                  libraries = ['blas', 'gsl']),
    ],
    include_dirs = [ np.get_include() ],
)
