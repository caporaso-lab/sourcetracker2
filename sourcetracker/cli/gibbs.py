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

import glob
import os
import click
import numpy as np
from functools import partial
import subprocess
import time
from biom import load_table
from biom.table import Table
from sourcetracker.cli import cli
from sourcetracker.sourcetracker import (parse_mapping_file, collapse_sources,
                                         sinks_and_sources,
                                         _cli_sync_biom_and_sample_metadata,
                                         _cli_collate_results, _cli_loo_runner,
                                         _cli_sink_source_prediction_runner,
                                         subsample_sources_sinks)
from ipyparallel import Client


@cli.command(name='gibbs')
@click.option('-i', '--table_fp', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to input BIOM table.')
@click.option('-m', '--mapping_fp', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to input category mapping file (sample metadata).')
@click.option('-o', '--output_dir', required=True,
              type=click.Path(exists=False, dir_okay=False),
              help='Path to the output directory to be created.')
@click.option('--loo', required=False, default=False, is_flag=True,
              help=('Classify each sample in `sources` using a leave-one-out '
                    'strategy. Replicates -s option in Knights et al. '
                    'sourcetracker.'))
@click.option('--jobs', required=False, default=1,
              type=click.INT,
              help='Number of processes to launch.')
@click.option('--alpha1', required=False, default=.001,
              type=click.FLOAT,
              help=('Prior counts of each species in the training '
                    'environments. Higher values decrease the trust in the '
                    'training environments, and make the source environment '
                    'distrubitons over taxa smoother. By default, this is '
                    'set to 0.001, which indicates reasonably high trust in '
                    'all source environments, even those with few training '
                    'sequences. This is useful when only a small number of '
                    'biological samples are available from a source '
                    'environment. A more conservative value would be 0.01.'))
@click.option('--alpha2', required=False, default=.001,
              type=click.FLOAT,
              help=('Prior counts of each species in Unknown environment. '
                    'Higher values make the Unknown environment smoother and '
                    'less prone to overfitting given a training sample.'))
@click.option('--beta', required=False, default=10,
              type=click.INT,
              help=('Count to be added to each species in each environment, '
                    'including `unknown`.'))
@click.option('--source_rarefaction_depth', required=False, default=1000,
              type=click.INT,
              help=('Depth at which to rarify sources. If 0, no '
                    'rarefaction performed.'))
@click.option('--sink_rarefaction_depth', required=False, default=1000,
              type=click.INT,
              help=('Depth at which to rarify sinks. If 0, no '
                    'rarefaction performed.'))
@click.option('--restarts', required=False, default=10,
              type=click.INT,
              help=('Number of independent Markov chains to grow. '
                    '`draws_per_restart` * `restarts` gives the number of '
                    'samplings of the mixing proportions that will be '
                    'generated.'))
@click.option('--draws_per_restart', required=False, default=1,
              type=click.INT,
              help=('Number of times to sample the state of the Markov chain '
                    'for each independent chain grown.'))
@click.option('--burnin', required=False, default=100,
              type=click.INT,
              help=('Number of passes (withdarawal and reassignment of every '
                    'sequence in the sink) that will be made before a sample '
                    '(draw) will be taken. Higher values allow more '
                    'convergence towards the true distribtion before draws '
                    'are taken.'))
@click.option('--delay', required=False, default=10,
              type=click.INT,
              help=('Number passes between each sampling (draw) of the '
                    'Markov chain. Once the burnin passes have been made, a '
                    'sample will be taken every `delay` number of passes. '
                    'This is also known as `thinning`. Thinning helps reduce '
                    'the impact of correlation between adjacent states of the '
                    'Markov chain.'))
@click.option('--cluster_start_delay', required=False, default=25,
              type=click.INT,
              help=('When using multiple jobs, the script has to start an '
                    '`ipcluster`. If ipcluster does not recognize that it '
                    'has been successfully started, the jobs will not be '
                    'successfully launched. If this is happening, increase '
                    'this parameter.'))
def gibbs(table_fp, mapping_fp, output_dir, loo, jobs, alpha1, alpha2, beta,
          source_rarefaction_depth, sink_rarefaction_depth,
          restarts, draws_per_restart, burnin, delay, cluster_start_delay):
    '''Gibb's sampler for Bayesian estimation of microbial sample sources.

    For details, see the project README file.
    '''
    # Create results directory. Click has already checked if it exists, and
    # failed if so.
    os.mkdir(output_dir)

    # Load the mapping file and biom table and remove samples which are not
    # shared.
    o = open(mapping_fp, 'U')
    sample_metadata_lines = o.readlines()
    o.close()

    sample_metadata, biom_table = \
        _cli_sync_biom_and_sample_metadata(
            parse_mapping_file(sample_metadata_lines),
            load_table(table_fp))

    # If biom table has fractional counts, it can produce problems in indexing
    # later on.
    biom_table.transform(lambda data, id, metadata: np.ceil(data))

    # If biom table has sample metadata, there will be pickling errors when
    # submitting multiple jobs. We remove the metadata by making a copy of the
    # table without metadata.
    biom_table = Table(biom_table._data.toarray(),
                       biom_table.ids(axis='observation'),
                       biom_table.ids(axis='sample'))

    # Parse the mapping file and options to get the samples requested for
    # sources and sinks.
    source_samples, sink_samples = sinks_and_sources(sample_metadata)

    # If we have no source samples neither normal operation or loo will work.
    # Will also likely get strange errors.
    if len(source_samples) == 0:
        raise ValueError('Mapping file or biom table passed contain no '
                         '`source` samples.')

    # Prepare the 'sources' matrix by collapsing the `source_samples` by their
    # metadata values.
    sources_envs, sources_data = collapse_sources(source_samples,
                                                  sample_metadata, 'Env',
                                                  biom_table, sort=True)

    # Rarefiy data if requested.
    sources_data, biom_table = \
        subsample_sources_sinks(sources_data, sink_samples, biom_table,
                                source_rarefaction_depth,
                                sink_rarefaction_depth)

    # Build function that require only a single parameter -- sample -- to
    # enable parallel processing if requested.
    if loo:
        f = partial(_cli_loo_runner, source_category='Env',
                    alpha1=alpha1, alpha2=alpha2, beta=beta,
                    restarts=restarts, draws_per_restart=draws_per_restart,
                    burnin=burnin, delay=delay,
                    sample_metadata=sample_metadata,
                    sources_data=sources_data, sources_envs=sources_envs,
                    biom_table=biom_table, output_dir=output_dir)
        sample_iter = source_samples
    else:
        f = partial(_cli_sink_source_prediction_runner, alpha1=alpha1,
                    alpha2=alpha2, beta=beta, restarts=restarts,
                    draws_per_restart=draws_per_restart, burnin=burnin,
                    delay=delay, sources_data=sources_data,
                    biom_table=biom_table, output_dir=output_dir)
        sample_iter = sink_samples

    if jobs > 1:
        # Launch the ipcluster and wait for it to come up.
        subprocess.Popen('ipcluster start -n %s --quiet' % jobs, shell=True)
        time.sleep(cluster_start_delay)
        c = Client()
        c[:].map(f, sample_iter, block=True)
        # Shut the cluster down. Answer taken from SO:
        # http://stackoverflow.com/questions/30930157/stopping-ipcluster-engines-ipython-parallel
        c.shutdown(hub=True)
    else:
        for sample in sample_iter:
            f(sample)

    # Format results for output.
    samples = []
    samples_data = []
    for sample_fp in glob.glob(os.path.join(output_dir, '*')):
        samples.append(sample_fp.strip().split('/')[-1].split('.txt')[0])
        samples_data.append(np.loadtxt(sample_fp, delimiter='\t'))
    mp, mps = _cli_collate_results(samples, samples_data, sources_envs)

    o = open(os.path.join(output_dir, 'mixing_proportions.txt'), 'w')
    o.writelines(mp)
    o.close()
    o = open(os.path.join(output_dir, 'mixing_proportions_stds.txt'), 'w')
    o.writelines(mps)
    o.close()
