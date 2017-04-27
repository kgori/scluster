#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='scluster',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "pandas"
    ],
    scripts=[
        "bin/sclust.py"
    ]
)
