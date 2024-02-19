#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import division

import click
import pandas as pd

from sourcetracker._cli import cli
from sourcetracker._plot import ST_graphs


@cli.command(name='plots')
@click.option('-m', '--mixing_props', required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-o', '--output_dir', required=True,
              type=click.Path(exists=False, dir_okay=True, file_okay=False,
                              writable=True))
@click.option('--stacked_bar', required=False, default=False, is_flag=True,
              show_default=True)
@click.option('--heatmap', required=False, default=True, is_flag=True,
              show_default=True)
@click.option('--paired_heatmap', required=False, default=False, is_flag=True,
              show_default=True)
@click.option('--title', required=False, default='Mixing Proportions',
              type=click.STRING, show_default=True)
@click.option('--heatmap_color', required=False, default='viridis',
              type=click.STRING, show_default=True)
@click.option('--unknowns', required=False, default=True, is_flag=True,
              show_default=True)
@click.option('--transpose', required=False, default=False, is_flag=True,
              show_default=True)
@click.option('--bar_color', required=False, default="", type=click.STRING,
              show_default=True)
@click.option('--flip_bar', required=False, default=False, is_flag=True,
              show_default=True)
@click.option('--x_lab', required=False, default="Sinks", type=click.STRING,
              show_default=True)
@click.option('--y_lab', required=False, default="Source Proportion",
              type=click.STRING, show_default=True)
def plots(
          mixing_props: pd.DataFrame,
          output_dir: str,
          title: str,
          stacked_bar: bool,
          heatmap: bool,
          paired_heatmap: bool,
          heatmap_color: str,
          unknowns: bool,
          transpose: bool,
          bar_color: str,
          flip_bar: bool,
          x_lab: str,
          y_lab: str):
    color_list = bar_color.split()
    graphs = ST_graphs(mixing_props, output_dir, title=title,
                       color=heatmap_color)
    if heatmap:
        graphs.ST_heatmap()
        if not unknowns:
            graphs.ST_heatmap(unknowns=False, ylabel=y_lab, xlabel=x_lab)
    if paired_heatmap:
        graphs.ST_paired_heatmap()
        if not unknowns:
            graphs.ST_paired_heatmap(unknowns=False, ylabel=y_lab)
            graphs.ST_paired_heatmap(unknowns=False, normalized=True,
                                     ylabel=y_lab)
        if transpose:
            graphs.ST_paired_heatmap(unknowns=False, normalized=True,
                                     transpose=True, ylabel=y_lab)
    if stacked_bar:
        graphs.ST_Stacked_bar(coloring=color_list, flipped=flip_bar,
                              x_lab=x_lab, y_lab=y_lab)
        if not unknowns:
            graphs.ST_Stacked_bar(unknowns=False, coloring=color_list,
                                  flipped=flip_bar, x_lab=x_lab,
                                  y_lab=y_lab)
