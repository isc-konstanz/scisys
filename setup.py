#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scisys
    ~~~~~~


"""
from os import path
from setuptools import setup, find_namespace_packages

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("scisys", "_version.py")) as f:
    exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'This repository provides a set of scientific processing functions for several ' \
              'energy system projects of ISC Konstanz.'

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    README = f.read()

NAME = 'scisys'
LICENSE = 'LGPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'adrian.minde@isc-konstanz.de'
URL = 'https://github.com/isc-konstanz/scisys'

INSTALL_REQUIRES = [
    'scipy >= 1.1',
    'tables >= 3.4',
    'openpyxl >= 2.4',
    'matplotlib >= 3',
    'seaborn >= 0.9',
    'corsys >= 0.8.4'
]

EXTRAS_REQUIRE = {
    'pdf': ['reportlab', 'svglib']
}

SCRIPTS = ['bin/scisys']

PACKAGES = find_namespace_packages(include=['scisys*'])

SETUPTOOLS_KWARGS = {
    'zip_safe': False,
    'include_package_data': True
}

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=README,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    url=URL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    scripts=SCRIPTS,
    **SETUPTOOLS_KWARGS
)
