#!/usr/bin/env python

from setuptools import find_packages, setup

readme = open('README.md').read()

requirements = [
    'torch',
    'torchvision',
]

setup(
    # Metadata
    name="torchsharp",
    version="0.0.1",
    author="Yusu Pan",
    author_email="xxdsox@gmail.com",
    description="Sharp Utilities for PyTorch.",
    long_description=readme,
    license="MIT",
    url="https://github.com/corenel/torchsharp",
    # Package info
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    install_requires=requirements,
)
