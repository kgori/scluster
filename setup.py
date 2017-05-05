#!/usr/bin/env python
#from numpy.distutils.core import Extension, setup
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='scluster',
    version='1.0.0',
    packages=['scluster'],
    install_requires=[
        "cython",
        "numpy",
        "pandas"
    ],
    scripts=[
        "bin/sclust.py"
    ],
    ext_modules=cythonize('src/evrot.pyx'),
    include_dirs=[numpy.get_include()]
)
