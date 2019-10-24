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

import pandas as pd
from biom import Table
from sourcetracker._sourcetracker import (intersect_and_sort_samples,
                                          get_samples, collapse_source_data,
                                          subsample_dataframe,
                                          validate_gibbs_input)
from sourcetracker._sourcetracker import gibbs as _gibbs
from sourcetracker._util import biom_to_df
# import default values
from sourcetracker._gibbs_defaults import (DEFAULT_ALPH1, DEFAULT_ALPH2,
                                           DEFAULT_TEN, DEFAULT_ONE,
                                           DEFAULT_HUND, DEFAULT_THOUS,
                                           DEFAULT_FLS, DEFAULT_SNK,
                                           DEFAULT_SRS, DEFAULT_SRS2,
                                           DEFAULT_CAT, DEFAULT_TRU)


def gibbs(feature_table: Table,
          sample_metadata: pd.DataFrame,
          loo: bool = DEFAULT_FLS,
          jobs: int = DEFAULT_ONE,
          alpha1: float = DEFAULT_ALPH1,
          alpha2: float = DEFAULT_ALPH2,
          beta: float = DEFAULT_TEN,
          source_rarefaction_depth: int = DEFAULT_THOUS,
          sink_rarefaction_depth: int = DEFAULT_THOUS,
          restarts: int = DEFAULT_TEN,
          draws_per_restart: int = DEFAULT_ONE,
          burnin: int = DEFAULT_HUND,
          delay: int = DEFAULT_ONE,
          per_sink_feature_assignments: bool = DEFAULT_TRU,
          sample_with_replacement: bool = DEFAULT_FLS,
          source_sink_column: str = DEFAULT_SNK,
          source_column_value: str = DEFAULT_SRS,
          sink_column_value: str = DEFAULT_SRS2,
          source_category_column: str = DEFAULT_CAT)\
              -> (pd.DataFrame, pd.DataFrame, Table, pd.DataFrame):
    # convert tables
    feature_table = biom_to_df(feature_table)
    sample_metadata = sample_metadata.to_dataframe()
    # run the gibbs sampler helper function (same used for q2)
    results = gibbs_helper(feature_table, sample_metadata, loo, jobs,
                           alpha1, alpha2, beta, source_rarefaction_depth,
                           sink_rarefaction_depth, restarts, draws_per_restart,
                           burnin, delay, per_sink_feature_assignments,
                           sample_with_replacement, source_sink_column,
                           source_column_value, sink_column_value,
                           source_category_column)
    # get the results (with fas)
    # here we only return the three df (via q2)
    mpm, mps, fas = results
    # make list filter

    def filter_list(inds, factor): return [ind for ind in list(inds)
                                           if ind not in factor]
    # concat each sink-source (dropping sources with same name as sink)
    fas_merged = pd.concat({sink: source.reindex(filter_list(source.index,
                                                             sink))
                            for sink, source in zip(mpm.columns, fas)})
    # if loo is True then columns are source-source
    if loo:
        columns_ = ['Source_one', 'Source_two']
    # if loo is False then columns as sink-source
    else:
        columns_ = ['Sink', 'Source']
    # make the index map and mapping in the same step
    ss_map = {'sample%i' % i: list(map(str, v))
              for i, v in enumerate(fas_merged.index.tolist())}
    ss_map = pd.DataFrame(ss_map, columns_).T
    ss_map.index.name = 'sampleid'
    # output for QIIME2
    fas_merged.index = ss_map.index
    fas_merged = Table(fas_merged.T.values,
                       fas_merged.T.index,
                       fas_merged.T.columns)
    # this is because QIIME will only
    # support these for now
    # in the future we will work
    # on supporting collections (i.e. fas)
    return mpm, mps, fas_merged, ss_map


def gibbs_helper(feature_table: Table,
                 sample_metadata: pd.DataFrame,
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
                 source_category_column: str) -> (pd.DataFrame,
                                                  pd.DataFrame,
                                                  list):
    '''Gibb's sampler for Bayesian estimation of microbial sample sources.

    This function is a helper that applies to both the click and QIIME2
    command line functionality.
    '''

    # Do high level check on feature data.
    feature_table = validate_gibbs_input(feature_table)

    # Remove samples not shared by both feature and metadata tables and order
    # rows equivalently.
    sample_metadata, feature_table = \
        intersect_and_sort_samples(sample_metadata, feature_table)

    # Identify source and sink samples.
    source_samples = get_samples(sample_metadata, source_sink_column,
                                 source_column_value)
    sink_samples = get_samples(sample_metadata, source_sink_column,
                               sink_column_value)

    # If we have no source samples neither normal operation or loo will work.
    # Will also likely get strange errors.
    if len(source_samples) == 0:
        raise ValueError(('You passed %s as the `source_sink_column` and %s '
                          'as the `source_column_value`. There are no samples '
                          'which are sources under these values. Please see '
                          'the help documentation and check your mapping '
                          'file.') % (source_sink_column, source_column_value))

    # Prepare the 'sources' matrix by collapsing the `source_samples` by their
    # metadata values.
    csources = collapse_source_data(sample_metadata, feature_table,
                                    source_samples, source_category_column,
                                    'mean')

    # Rarify collapsed source data if requested.
    if source_rarefaction_depth > 0:
        d = (csources.sum(1) >= source_rarefaction_depth)
        if not d.all():
            count_too_shallow = (~d).sum()
            shallowest = csources.sum(1).min()
            raise ValueError(('You requested rarefaction of source samples at '
                              '%s, but there are %s collapsed source samples '
                              'that have less sequences than that. The '
                              'shallowest of these is %s sequences.') %
                             (source_rarefaction_depth, count_too_shallow,
                              shallowest))
        else:
            csources = subsample_dataframe(csources, source_rarefaction_depth,
                                           replace=sample_with_replacement)

    # Prepare to rarify sink data if we are not doing LOO. If we are doing loo,
    # we skip the rarefaction, and set sinks to `None`.
    if not loo:
        sinks = feature_table.loc[sink_samples, :]
        if sink_rarefaction_depth > 0:
            d = (sinks.sum(1) >= sink_rarefaction_depth)
            if not d.all():
                count_too_shallow = (~d).sum()
                shallowest = sinks.sum(1).min()
                raise ValueError(('You requested rarefaction of sink samples '
                                  'at %s, but there are %s sink samples that '
                                  'have less sequences than that. The '
                                  'shallowest of these is %s sequences.') %
                                 (sink_rarefaction_depth, count_too_shallow,
                                  shallowest))
            else:
                sinks = subsample_dataframe(sinks, sink_rarefaction_depth,
                                            replace=sample_with_replacement)
    else:
        sinks = None

    # Run the computations.
    mpm, mps, fas = _gibbs(csources, sinks, alpha1, alpha2, beta, restarts,
                           draws_per_restart, burnin, delay, jobs,
                           create_feature_tables=per_sink_feature_assignments)
    # number of returns chnages based on flag
    # this was refactored for QIIME2
    # transpose to follow convention
    # rows are features (i.e. taxa)
    # columns are samples
    if per_sink_feature_assignments:
        return mpm.T, mps.T, fas
    else:
        return mpm.T, mps.T
