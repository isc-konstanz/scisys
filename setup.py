#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    th-e-data
    ~~~~~~~~~


"""
from os import path
from setuptools import setup, find_namespace_packages

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("th_e_data", "_version.py")) as f:
    exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'This repository provides a set of data processing functions for several projects of ISC Konstanz.'

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    README = f.read()

NAME = 'th-e-data'
LICENSE = 'LGPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'adrian.minde@isc-konstanz.de'
URL = 'https://github.com/isc-konstanz/th-e-data'

INSTALL_REQUIRES = [
    'tables >= 3.4',
    "th-e-core >= 0.6 @ git+https://github.com/isc-konstanz/th-e-core.git@master"
]

EXTRA_REQUIRES = {
    'openpyxl >= 2.4': ['xlsx', 'excel'],
    'seaborn >= 0.9': ['plot'],
    'matplotlib >= 3': ['plot']
}

SCRIPTS = ['bin/th-e-data']

PACKAGES = find_namespace_packages(include=['th_e_data*'])

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
    extras_require=EXTRA_REQUIRES,
    scripts=SCRIPTS,
    **SETUPTOOLS_KWARGS
)
