from setuptools import setup, Extension
from Cython.Distutils import build_ext

setup(
    name = 'neural',
    version = '0.1',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension('neural._util', ['neural/_util.pyx']),
    ],
)
