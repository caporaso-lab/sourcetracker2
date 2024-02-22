# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import click

from importlib import import_module
from sourcetracker import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass

import_module('sourcetracker._cli.gibbs')
import_module('sourcetracker._cli.plots')
