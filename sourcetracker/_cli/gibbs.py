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

import os
import click
import pandas as pd
from biom import Table, load_table
from sourcetracker._cli import cli
from sourcetracker._gibbs import gibbs_helper
from sourcetracker._plot import plot_heatmap
from sourcetracker._util import parse_sample_metadata

# import default descriptions
from sourcetracker._gibbs_defaults import (DESC_TBL, DESC_MAP, DESC_OUT,
                                           DESC_LOO, DESC_JBS, DESC_ALPH1,
                                           DESC_ALPH2, DESC_BTA, DESC_RAF1,
                                           DESC_RAF2, DESC_RST, DESC_DRW,
                                           DESC_BRN, DESC_DLY, DESC_PFA,
                                           DESC_RPL, DESC_SNK, DESC_SRS,
                                           DESC_SRS2, DESC_CAT)

# import default values
from sourcetracker._gibbs_defaults import (DEFAULT_ALPH1, DEFAULT_ALPH2,
                                           DEFAULT_TEN, DEFAULT_ONE,
                                           DEFAULT_HUND, DEFAULT_THOUS,
                                           DEFAULT_FLS, DEFAULT_SNK,
                                           DEFAULT_SRS, DEFAULT_SRS2,
                                           DEFAULT_CAT)


@cli.command(name='gibbs')
@click.option('-i', '--table_fp', required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help=DESC_TBL)
@click.option('-m', '--mapping_fp', required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help=DESC_MAP)
@click.option('-o', '--output_dir', required=True,
              type=click.Path(exists=False, dir_okay=True, file_okay=False,
                              writable=True),
              help=DESC_OUT)
@click.option('--loo', required=False, default=DEFAULT_FLS, is_flag=True,
              show_default=True,
              help=DESC_LOO)
@click.option('--jobs', required=False, default=DEFAULT_ONE,
              type=click.INT, show_default=True,
              help=DESC_JBS)
@click.option('--alpha1', required=False, default=DEFAULT_ALPH1,
              type=click.FLOAT, show_default=True,
              help=DESC_ALPH1)
@click.option('--alpha2', required=False, default=DEFAULT_ALPH2,
              type=click.FLOAT, show_default=True,
              help=DESC_ALPH2)
@click.option('--beta', required=False, default=DEFAULT_TEN,
              type=click.FLOAT, show_default=True,
              help=DESC_BTA)
@click.option(
    '--source_rarefaction_depth',
    required=False,
    default=DEFAULT_THOUS,
    type=click.IntRange(
        min=0,
        max=None),
    show_default=True,
    help=DESC_RAF1)
@click.option(
    '--sink_rarefaction_depth',
    required=False,
    default=DEFAULT_THOUS,
    type=click.IntRange(
        min=0,
        max=None),
    show_default=True,
    help=DESC_RAF2)
@click.option('--restarts', required=False, default=DEFAULT_TEN,
              type=click.INT, show_default=True,
              help=DESC_RST)
@click.option('--draws_per_restart', required=False, default=DEFAULT_ONE,
              type=click.INT, show_default=True,
              help=DESC_DRW)
@click.option('--burnin', required=False, default=DEFAULT_HUND,
              type=click.INT, show_default=True,
              help=DESC_BRN)
@click.option('--delay', required=False, default=DEFAULT_ONE,
              type=click.INT, show_default=True,
              help=DESC_DLY)
@click.option(
    '--per_sink_feature_assignments',
    required=False,
    default=DEFAULT_FLS,
    is_flag=True,
    show_default=True,
    help=DESC_PFA)
@click.option('--sample_with_replacement', required=False,
              default=DEFAULT_FLS, show_default=True, is_flag=True,
              help=DESC_RPL)
@click.option('--source_sink_column', required=False, default=DEFAULT_SNK,
              type=click.STRING, show_default=True,
              help=DESC_SNK)
@click.option('--source_column_value', required=False, default=DEFAULT_SRS,
              type=click.STRING, show_default=True,
              help=DESC_SRS)
@click.option('--sink_column_value', required=False, default=DEFAULT_SRS2,
              type=click.STRING, show_default=True,
              help=DESC_SRS2)
@click.option('--source_category_column', required=False, default=DEFAULT_CAT,
              type=click.STRING, show_default=True,
              help=DESC_CAT)
def gibbs(table_fp: Table,
          mapping_fp: pd.DataFrame,
          output_dir: str,
          loo: bool,
          jobs: int,
          alpha1: float,
          alpha2: float,
          beta: float,
          source_rarefaction_depth: int,
          sink_rarefaction_depth: int,
          restarts: int,
          draws_per_restart: int,
          burnin: int,
          delay: int,
          per_sink_feature_assignments: bool,
          sample_with_replacement: bool,
          source_sink_column: str,
          source_column_value: str,
          sink_column_value: str,
          source_category_column: str):
    '''Gibb's sampler for Bayesian estimation of microbial sample sources.

    For details, see the project README file.
    '''
    # Create results directory. Click has already checked if it exists, and
    # failed if so.
    os.mkdir(output_dir)

    # Load the metadata file and feature table.
    sample_metadata = parse_sample_metadata(open(mapping_fp, 'U'))
    feature_table = load_table(table_fp)

    # run the gibbs sampler helper function (same used for q2)
    results = gibbs_helper(feature_table, sample_metadata, loo, jobs,
                           alpha1, alpha2, beta, source_rarefaction_depth,
                           sink_rarefaction_depth, restarts, draws_per_restart,
                           burnin, delay, per_sink_feature_assignments,
                           sample_with_replacement, source_sink_column,
                           source_column_value, sink_column_value,
                           source_category_column)
    # import the results (will change based on per_sink_feature_assignments)
    if len(results) == 3:
        mpm, mps, fas = results
        # write the feature tables from fas
        for sink, fa in zip(mpm.columns, fas):
            fa.to_csv(os.path.join(output_dir, sink + '.feature_table.txt'),
                      sep='\t')
    else:
        # get the results (without fas)
        mpm, mps = results

    # Write results.
    mpm.to_csv(os.path.join(output_dir, 'mixing_proportions.txt'), sep='\t')
    mps.to_csv(os.path.join(output_dir, 'mixing_proportions_stds.txt'),
               sep='\t')

    # Plot contributions.
    fig, ax = plot_heatmap(mpm.T)
    fig.savefig(os.path.join(output_dir, 'mixing_proportions.pdf'), dpi=300)
