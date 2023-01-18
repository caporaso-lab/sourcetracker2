#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools.command.egg_info import egg_info
from setuptools.command.develop import develop
from setuptools.command.install import install
import os
from setuptools import find_packages, setup

__version__ = '2.0.1-dev'

# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.c'


# bootstrap numpy intall
# https://stackoverflow.com/questions/51546255/
# python-package-setup-setup-py-with-customisation
# -to-handle-wrapped-fortran
def custom_command():
    import sys
    if sys.platform in ['darwin', 'linux']:
        os.system('pip install numpy')


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


extensions = [
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

classes = """
    Development Status :: 1 - Planning
    Intended Audience :: Science/Research
    Natural Language :: English
    Operating System :: MacOS :: MacOS X
    Operating System :: POSIX
    Operating System :: Unix
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = "Python implementation of the SourceTracker R package."

standalone = ['sourcetracker2=sourcetracker._cli.gibbs:gibbs']
q2cmds = ['q2-sourcetracker2=sourcetracker._q2.plugin_setup:plugin']

with open('README.md') as f:
    long_description = f.read()


setup(
    name='sourcetracker',
    version=__version__,
    license='modified BSD',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Biota Technology',
    author_email='will@biota.com',
    maintainer='Will Van Treuren',
    maintainer_email='will@biota.com',
    url='http://www.biota.com',
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=[
          'numpy',
          'click',
          'pandas',
          'scipy',
          'nose',
          'scikit-learn',
          'scikit-bio',
          'biom-format',
          'h5py',
          'seaborn'],
    classifiers=classifiers,
    package_data={'sourcetracker': ['citations.bib']},
    entry_points={'qiime2.plugins': q2cmds,
                  'console_scripts': standalone},
    cmdclass={'install': CustomInstallCommand,
              'develop': CustomDevelopCommand,
              'egg_info': CustomEggInfoCommand},
    zip_safe=False)
