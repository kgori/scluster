#!/usr/bin/env python
from numpy.distutils.core import Extension, setup

extensions = [Extension('evrot_extensions', ['src/evrot_extensions.c'])]

setup(
    name='scluster',
    version='1.0.0',
    packages=['scluster'],
    install_requires=[
        "numpy",
        "pandas"
    ],
    scripts=[
        "bin/sclust.py"
    ],
    ext_modules=extensions
)
