# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from importlib import import_module
import click

from sourcetracker import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass

import_module('sourcetracker.cli.gibbs')
