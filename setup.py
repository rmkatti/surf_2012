from setuptools import setup, Extension
from Cython.Distutils import build_ext

setup(
    name = 'neural',
    version = '0.1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension('neural.external.tokyo', ['neural/external/tokyo.pyx'],
                  libraries = ['blas']),
        Extension('neural._helmholtz', ['neural/_helmholtz.pyx']),
        Extension('neural._util', ['neural/_util.pyx'],
                  libraries = ['blas', 'gsl']),
    ],
)
