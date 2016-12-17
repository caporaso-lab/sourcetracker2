#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial


def compare_sink_metrics():
    return list(sorted(_compare_sinks_metrics.keys()))


def compare_sinks(observed, expected, metric):
    """ Compare the mix proportions for a collection of sinks

    Parameters
    ----------
    observed: pd.DataFrame
        Observed mix proportions where sinks are indices and sources are
        columns.
    observed: pd.DataFrame
        Expected mix proportions where sinks are indices and sources are
        columns.
    metric: str
        The metric to use for comparing sinks.

    Returns
    -------
    pd.DataFrame
        Comparison results where indices are sinks and columns indicate
        similarity or difference, depending on the metric. The specific
        columns that are included will be metric-dependent.

    See Also
    --------
    sourcetracker._compare.compare_sink_metrics

    """

    if metric not in _compare_sinks_metrics:
        raise KeyError("%s is not a known metric. Known metrics are: %s."
                       % (metric, ', '.join(_compare_sinks_metrics)))
    else:
        metric_fn = _compare_sinks_metrics[metric]

    _validate_dataframes(observed, expected)

    # Depending on the compare function being called, these sorts won't always
    # be necessary, but it's a nice catch to have since we then don't need to
    # confirm that all compare functions do this. I expect that these should
    # be quick since the size of results being compared likely isn't very big
    # but we can revisit if this is a bottleneck.
    return metric_fn(observed.sort_index(), expected.sort_index())


def _validate_dataframes(observed, expected):
    """ Confirm index and columns contain the same values
    """
    if set(observed.index) != set(expected.index):
        raise ValueError('Sinks in observed and expected results must be '
                         'identical.')

    if set(observed.columns) != set(expected.columns):
        raise ValueError('Sources in observed and expected results must be '
                         'identical.')


def _spearman(observed, expected):
    results = []
    for id_ in observed.index:
        rho, p = scipy.stats.spearmanr(observed.loc[id_], expected.loc[id_])
        results.append((rho, p))
    return pd.DataFrame(results, index=observed.index,
                        columns=['Spearman rho', 'p'])


def _pearson(observed, expected):
    results = []
    for id_ in observed.index:
        rho, p = scipy.stats.pearsonr(observed.loc[id_], expected.loc[id_])
        results.append((rho, p))
    return pd.DataFrame(results, index=observed.index,
                        columns=['Pearson r', 'p'])


def _euclidean(observed, expected):
    results = []
    for id_ in observed.index:
        d = scipy.spatial.distance.euclidean(observed.loc[id_],
                                             expected.loc[id_])
        results.append(d)
    return pd.DataFrame(results, index=observed.index,
                        columns=['Euclidean distance'])


def _absolute_difference(observed, expected):
    results = []
    for id_ in observed.index:
        results.append(np.abs(observed.loc[id_] - expected.loc[id_]))
    return pd.DataFrame(results, index=observed.index,
                        columns=observed.columns)


_compare_sinks_metrics = {'spearman': _spearman, 'pearson': _pearson,
                          'euclidean': _euclidean,
                          'absolute_difference': _absolute_difference}
