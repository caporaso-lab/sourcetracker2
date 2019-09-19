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

from biom import load_table
from sourcetracker._cli import cli
from sourcetracker._sourcetracker import (gibbs, intersect_and_sort_samples,
                                          get_samples, collapse_source_data,
                                          subsample_dataframe,
                                          validate_gibbs_input)
from sourcetracker._gibbs import gibbs_helper
from sourcetracker._util import parse_sample_metadata, biom_to_df
from sourcetracker._plot import plot_heatmap


@cli.command(name='gibbs')
@click.option('-i', '--table_fp', required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help='Path to input BIOM table.')
@click.option('-m', '--mapping_fp', required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help='Path to sample metadata mapping file.')
@click.option('-o', '--output_dir', required=True,
              type=click.Path(exists=False, dir_okay=True, file_okay=False,
                              writable=True),
              help='Path to the output directory to be created.')
@click.option('--loo', required=False, default=False, is_flag=True,
              show_default=True,
              help=('Classify each sample in `sources` using a leave-one-out '
                    'strategy. Replicates -s option in Knights et al. '
                    'sourcetracker.'))
@click.option('--jobs', required=False, default=1,
              type=click.INT, show_default=True,
              help='Number of processes to launch.')
@click.option('--alpha1', required=False, default=.001,
              type=click.FLOAT, show_default=True,
              help=('Prior counts of each feature in the training '
                    'environments. Higher values decrease the trust in the '
                    'training environments, and make the source environment '
                    'distributions over taxa smoother. A value of 0.001 '
                    'indicates reasonably high trust in all source '
                    'environments, even those with few training sequences. A '
                    'more conservative value would be 0.01.'))
@click.option('--alpha2', required=False, default=.1,
              type=click.FLOAT, show_default=True,
              help=('Prior counts of each feature in the `unknown` environment'
                    ' as a fraction of the counts of the current sink being '
                    'evaluated. Higher values make the `unknown` environment '
                    'smoother and less prone to overfitting given a training '
                    'sample.'))
@click.option('--beta', required=False, default=10,
              type=click.FLOAT, show_default=True,
              help=('Count to be added to each feature in each environment, '
                    'including `unknown` for `p_v` calculations.'))
@click.option('--source_rarefaction_depth', required=False, default=1000,
              type=click.IntRange(min=0, max=None), show_default=True,
              help=('Depth at which to rarify sources. If 0, no '
                    'rarefaction performed.'))
@click.option('--sink_rarefaction_depth', required=False, default=1000,
              type=click.IntRange(min=0, max=None), show_default=True,
              help=('Depth at which to rarify sinks. If 0, no '
                    'rarefaction performed.'))
@click.option('--restarts', required=False, default=10,
              type=click.INT, show_default=True,
              help=('Number of independent Markov chains to grow. '
                    '`draws_per_restart` * `restarts` gives the number of '
                    'samplings of the mixing proportions that will be '
                    'generated.'))
@click.option('--draws_per_restart', required=False, default=1,
              type=click.INT, show_default=True,
              help=('Number of times to sample the state of the Markov chain '
                    'for each independent chain grown.'))
@click.option('--burnin', required=False, default=100,
              type=click.INT, show_default=True,
              help=('Number of passes (withdarawal and reassignment of every '
                    'sequence in the sink) that will be made before a sample '
                    '(draw) will be taken. Higher values allow more '
                    'convergence towards the true distribtion before draws '
                    'are taken.'))
@click.option('--delay', required=False, default=1,
              type=click.INT, show_default=True,
              help=('Number passes between each sampling (draw) of the '
                    'Markov chain. Once the burnin passes have been made, a '
                    'sample will be taken, and then taken again every `delay` '
                    'number of passes. This is also known as `thinning`. '
                    'Thinning helps reduce the impact of correlation between '
                    'adjacent states of the Markov chain.'))
@click.option('--per_sink_feature_assignments', required=False, default=False,
              is_flag=True, show_default=True,
              help=('If True, this option will cause SourceTracker2 to write '
                    'out a feature table for each sink (or source if `--loo` '
                    'is passed). These feature tables contain the specific '
                    'sequences that contributed to a sink from a given '
                    'source. This option can be memory intensive if there are '
                    'a large number of features.'))
@click.option('--sample_with_replacement', required=False,
              default=False, show_default=True, is_flag=True,
              help=('Sample with replacement instead of '
                    'sample without replacement'))
@click.option('--source_sink_column', required=False, default='SourceSink',
              type=click.STRING, show_default=True,
              help=('Sample metadata column indicating which samples should be'
                    ' treated as sources and which as sinks.'))
@click.option('--source_column_value', required=False, default='source',
              type=click.STRING, show_default=True,
              help=('Value in source_sink_column indicating which samples '
                    'should be treated as sources.'))
@click.option('--sink_column_value', required=False, default='sink',
              type=click.STRING, show_default=True,
              help=('Value in source_sink_column indicating which samples '
                    'should be treated as sinks.'))
@click.option('--source_category_column', required=False, default='Env',
              type=click.STRING, show_default=True,
              help=('Sample metadata column indicating the type of each '
                    'source sample.'))
def gibbs_cli(table_fp, mapping_fp, output_dir, loo, jobs, alpha1, alpha2,
              beta, source_rarefaction_depth, sink_rarefaction_depth, restarts,
              draws_per_restart, burnin, delay, per_sink_feature_assignments,
              sample_with_replacement, source_sink_column,
              source_column_value, sink_column_value,
              source_category_column):
    '''Gibb's sampler for Bayesian estimation of microbial sample sources.

    For details, see the project README file.
    '''
    # Create results directory. Click has already checked if it exists, and
    # failed if so.
    os.mkdir(output_dir)

    # run the gibbs sampler helper function (same used for q2)
    mpm, mps, fas =  gibbs_helper(table_fp, mapping_fp, loo, jobs,
                                  alpha1, alpha2, beta, source_rarefaction_depth,
                                  sink_rarefaction_depth, restarts, draws_per_restart,
                                  burnin, delay, per_sink_feature_assignments, 
                                  sample_with_replacement, source_sink_column,
                                  source_column_value, sink_column_value,
                                  source_category_column)

    # Write results.
    mpm.to_csv(os.path.join(output_dir, 'mixing_proportions.txt'), sep='\t')
    mps.to_csv(os.path.join(output_dir, 'mixing_proportions_stds.txt'),
               sep='\t')
    if per_sink_feature_assignments:
        for sink, fa in zip(mpm.index, fas):
            fa.to_csv(os.path.join(output_dir, sink + '.feature_table.txt'),
                      sep='\t')

    # Plot contributions.
    fig, ax = plot_heatmap(mpm)
    fig.savefig(os.path.join(output_dir, 'mixing_proportions.pdf'), dpi=300)