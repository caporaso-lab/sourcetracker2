#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import find_packages, setup

__version__ = '2.0.1-dev'

classes = """
    Development Status :: 1 - Planning
    Intended Audience :: Science/Research
    Natural Language :: English
    Operating System :: MacOS :: MacOS X
    Operating System :: POSIX
    Operating System :: Unix
    Programming Language :: Python
    Programming Language :: Python :: 3.5
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = "Python implementation of the SourceTracker R package."

setup(
    name='sourcetracker',
    version=__version__,
    license='modified BSD',
    description=description,
    long_description=description,
    author='Biota Technology',
    author_email='will@biota.com',
    maintainer='Will Van Treuren',
    maintainer_email='will@biota.com',
    url='http://www.biota.com',
    test_suite='nose.collector',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'biom-format >= 2.1.5, < 2.2.0',
                      'click', 'ipyparallel', 'scikit-bio', 'pandas',
                      'scipy >= 0.18.0'],
    classifiers=classifiers,
    entry_points='''
        [console_scripts]
        sourcetracker2=sourcetracker._cli:cli
        ''')
