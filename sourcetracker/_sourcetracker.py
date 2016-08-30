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

import numpy as np
from skbio.stats import subsample_counts
import pandas as pd
from functools import partial


def check_and_correct_data(feature_table, apply_fractional_value_correction):
    '''Check and correct problems in the `feature_table`.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table where rows are features and columns are samples.
    apply_fractional_value_correction : boolean
        Fractional counts cause errors in indexing. Uses np.ceil to remove
        them if True.

    Returns
    -------
    feature_table : pd.DataFrame
        Corrected feature table if changes have been made or original table if
        no changes occurred. Will automatically be cast as dtype=np.int64

    Raises
    ------
    ValueError
        If data contains values that Gibb's sampling cannot handle.
    '''
    if not np.isreal(feature_table.dtypes).all():
        raise TypeError('Feature table contains one or more columns which are '
                        'not numeric type. This is likely due to boolean, '
                        'string, or other data types being present. Data must '
                        'contain exclusively real-valued integers or floats.')

    if np.isnan(feature_table.values).any():
        raise ValueError('One or more values in the feature table is a `nan` '
                         'or `null` value. Data must contain exclusively real-'
                         'valued integers or floats.')

    if not (np.ceil(feature_table.values) == feature_table.values).all():
        if apply_fractional_value_correction is True:
            return pd.DataFrame(np.ceil(feature_table.values).astype(np.int64),
                                index=feature_table.index,
                                columns=feature_table.columns)
        else:
            raise ValueError('Non-integer data in the feature table is not '
                             'being corrected by `check_and_correct_data`. '
                             'Data leaving this function must be np.int64 '
                             'type. Either correct the table or pass the '
                             '`apply_fractional_value_correction` to '
                             '`check_and_correct_data`.')
    return feature_table.astype(np.int64)


def biom_to_df(biom_table, apply_fractional_value_correction=True):
    '''Turn biom table into dataframe, correcting fractional counts.

    Parameters
    ----------
    biom_table : biom.table.Table
        Biom table.
    apply_fractional_value_correction : boolean
        Fractional counts cause errors in indexing. Uses np.ceil to remove
        them if True.

    Returns
    -------
    feature_table : pd.DataFrame
        Contingency table with rows, columns = samples, features.
    '''
    table = pd.DataFrame(biom_table._data.toarray().T,
                         index=biom_table.ids(axis='sample'),
                         columns=biom_table.ids(axis='observation'))
    return check_and_correct_data(table, apply_fractional_value_correction)


def intersect_and_sort_samples(sample_metadata, feature_table):
    '''Return input tables retaining only shared samples, row order equivalent.

    Parameters
    ----------
    sample_metadata : pd.DataFrame
        Contingency table with rows, columns = samples, metadata.
    feature_table : pd.DataFrame
        Contingency table with rows, columns = samples, features.

    Returns
    -------
    sample_metadata, feature_table : pd.DataFrame, pd.DataFrame
        Input tables with unshared samples removed and ordered equivalently.

    Raises
    ------
    ValueError
        If no shared samples are found.
    '''
    shared_samples = np.intersect1d(sample_metadata.index, feature_table.index)
    if shared_samples.size == 0:
        raise ValueError('There are no shared samples between the feature '
                         'table and the sample metadata. Ensure that you have '
                         'passed the correct files.')
    elif (shared_samples.size == sample_metadata.shape[0] ==
          feature_table.shape[0]):
        s_metadata = sample_metadata.copy()
        s_features = feature_table.copy()
    else:
        s_metadata = sample_metadata.loc[np.in1d(sample_metadata.index,
                                                 shared_samples), :].copy()
        s_features = feature_table.loc[np.in1d(feature_table.index,
                                               shared_samples), :].copy()
    return s_metadata, s_features.loc[s_metadata.index, :]


def get_samples(sample_metadata, col, value):
    '''Return samples which have `value` under `col`.'''
    return sample_metadata.index[sample_metadata[col] == value].copy()


def collapse_source_data(sample_metadata, feature_table, source_samples,
                         category, method):
    '''Collapse each set of source samples into an aggregate source.

    Parameters
    ----------
    sample_metadata : pd.DataFrame
        Contingency table where rows are features and columns are metadata.
    feature_table : pd.DataFrame
        Contingency table where rows are features and columns are samples.
    source_samples : iterable
        Samples which should be considered for collapsing (i.e. are sources).
    category : str
        Column in `sample_metadata` which should be used to group samples.
    method : str
        One of the available aggregation methods in pd.DataFrame.agg (mean,
        median, prod, sum, std, var).

    Returns
    -------
    pd.DataFrame
        Collapsed sample data.

    Notes
    -----
    This function calls `check_and_correct_data` before returning the collapsed
    source table. This is required in case the aggregation function causes
    nans or non-integer data to be returned.

    The order of the collapsed sources is determined by the sort order of their
    names. For instance, in the example below, .4 comes before 3.0 so the
    collapsed sources will have the 0th row as .4.

    Examples
    --------
    >>> samples = ['sample1', 'sample2', 'sample3', 'sample4']
    >>> category = 'pH'
    >>> values = [3.0, 0.4, 3.0, 3.0]
    >>> stable = pd.DataFrame(values, index=samples, columns = [category])
    >>> stable
                   pH
        sample1   3.0
        sample2   0.4
        sample3   3.0
        sample4   3.0

    >>> fdata = np.array([[ 10,  50,  10,  70],
                          [  0,  25,  10,   5],
                          [  0,  25,  10,   5],
                          [100,   0,  10,   5]])
    >>> ftable = pd.DataFrame(fdata, index = stable.index)
    >>> ftable
               0   1   2   3
    sample1   10  50  10  70
    sample2    0  25  10   5
    sample3    0  25  10   5
    sample4  100   0  10   5

    >>> source_samples = ['sample1', 'sample2', 'sample3']
    >>> method = 'sum'
    >>> csources = collapse_source_data(stable, ftable, source_samples,
                                        category, method)
    >>> csources
                   0   1   2   3
    collapse_col
    0.4            0  25  10   5
    3.0           10  75  20  75
    '''
    sources = sample_metadata.loc[source_samples, :]
    table = feature_table.loc[sources.index, :].copy()
    table['collapse_col'] = sources[category]
    return check_and_correct_data(table.groupby('collapse_col').agg(method),
                                  True)


def subsample_dataframe(df, depth):
    '''Subsample (rarify) input dataframe without replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table where rows are features and columns are samples.
    depth : int
        Number of sequences to choose per sample.

    Returns
    -------
    pd.DataFrame
        Subsampled dataframe.
    '''
    f = partial(subsample_counts, n=depth, replace=False)
    return df.apply(f, axis=1, reduce=False, raw=False)


def generate_environment_assignments(n, num_sources):
    '''Randomly assign `n` counts to one of `num_sources` environments.

    Parameters
    ----------
    n : int
        Number of environment assignments to generate.
    num_sources : int
        Number of possible environment states (this includes the 'Unknown').

    Returns
    -------
    seq_env_assignments : np.array
        1D vector of length `n`. The ith entry is the environment assignment of
        the ith feature.
    envcounts : np.array
        1D vector of length `num_sources`. The ith entry is the total number of
        entries in `seq_env_assignments` which are equal to i.
    '''
    seq_env_assignments = np.random.choice(np.arange(num_sources), size=n,
                                           replace=True)
    envcounts = np.bincount(seq_env_assignments, minlength=num_sources)
    return seq_env_assignments, envcounts


class ConditionalProbability(object):
    def __init__(self, alpha1, alpha2, beta, source_data):
        r"""Set properties used for calculating the conditional probability.

        Paramaters
        ----------
        alpha1 : float
            Prior counts of each feature in the training environments. Higher
            values decrease the trust in the training environments, and make
            the source environment distributions over taxa smoother. By
            default, this is set to 0.001, which indicates reasonably high
            trust in all source environments, even those with few training
            sequences. This is useful when only a small number of biological
            samples are available from a source environment. A more
            conservative value would be 0.01.
        alpha2 : float
            Prior counts of each feature in the Unknown environment. Higher
            values make the Unknown environment smoother and less prone to
            overfitting given a training sample.
        beta : float
            Number of prior counts of test sequences from each feature in each
            environment
        source_data : np.array
            Columns are features, rows are collapsed samples. The [i,j]
            entry is the sum of the counts of features j in all samples which
            were considered part of source i.

        Attributes
        ----------
        m_xivs : np.array
            This is an exact copy of the source_data passed when the function
            is initialized. It is referenced as m_xivs because m_xiv is the
            [v, xi] entry of the source data. In other words, the count of the
            xith feature in the vth environment.
        m_vs : np.array
            The row sums of self.m_xivs. This is referenced as m_v in [1]_.
        V : int
            Number of environments (includes both known sources and the
            'unknown' source).
        tau : int
            Number of features.
        joint_probability : np.array
            The joint conditional distribution. Until the `precalculate` method
            is called, this will be uniformly zero.
        n : int
            Number of sequences in the sink.
        known_p_tv : np.array
            An array giving the precomputable parts of the probability of
            finding the xith taxon in the vth environment given the known
            sources, aka p_tv in the R implementation. Rows are (known)
            sources, columns are features, shape is (V-1, tau).
        denominator_p_v : float
            The denominator of the calculation for finding the probability of
            a sequence being in the vth environment given the training data
            (source data).
        known_source_cp : np.array
            All precomputable portions of the conditional probability array.
            Dimensions are the same as self.known_p_tv.

        Notes
        -----
        This class exists to calculate the conditional probability given in
        reference [1]_ (with modifications based on communications with the
        author). Since the calculation of the conditional probability must
        occur during each pass of the Gibbs sampler, reducing the number of
        computations is of paramount importance. This class precomputes
        everything that is static throughout a run of the sampler to reduce the
        innermost for-loop computations.

        The formula used to calculate the conditional joint probability is
        described in the project readme file.

        The variables are named in the class, as well as its methods, in
        accordance with the variable names used in [1]_.

        Examples
        --------
        The class is written so that it will be created before being passed to
        the function which handles the loops of the Gibbs sampling.
        >>> cp = ConditionalProbability(alpha1 = .5, alpha2 = .001, beta = 10,
        ...                             np.array([[0, 0, 0, 100, 100, 100],
        ...                                      [100, 100, 100, 0, 0, 0]]))
        Once it is passed to the Gibbs sampling function, the number of
        sequences in the sink becomes known, and we can update the object with
        this information to allow final precomputation.
        >>> cp.set_n(367)
        >>> cp.precompute()
        Now we can compute the 'slice' of the conditional probability depending
        on the current state of the test sequences (the ones randomly assigned
        and then iteratively reassigned) and which feature (the slice) the
        sequence we have removed was from.
        >>> xi = 2
        Count of the training sequences (that are feature xi) currently
        assigned to the unknown environment.
        >>> m_xiV = 38
        Sum of the training sequences currently assigned to the unknown
        environment (over all features).
        >>> m_V = 158
        Counts of the test sequences in each environment at the current
        iteration of the sampler.
        >>> n_vnoti = np.array([10, 500, 6])
        Calculating the probability slice.
        >>> cp.calculate_cp_slice(xi, m_xiV, m_V, n_vnoti)
        array([8.55007781e-05, 4.38234238e-01, 9.92823532e-03])

        References
        ----------
        .. [1] Knights et al. "Bayesian community-wide culture-independent
           source tracking", Nature Methods 2011.
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.m_xivs = source_data.astype(np.float64)
        self.m_vs = np.expand_dims(source_data.sum(1),
                                   axis=1).astype(np.float64)
        self.V = source_data.shape[0] + 1
        self.tau = source_data.shape[1]
        # Create the joint probability vector which will be overwritten each
        # time self.calculate_cp_slice is called.
        self.joint_probability = np.zeros(self.V, dtype=np.float64)

    def set_n(self, n):
        """Set the sum of the sink."""
        self.n = n

    def precalculate(self):
        """Precompute all static quantities of the probability matrix."""
        # Known source.
        self.known_p_tv = (self.m_xivs + self.alpha1) / \
                          (self.m_vs + self.tau * self.alpha1)
        self.denominator_p_v = self.n - 1 + (self.beta * self.V)

        # We are going to be accessing columns of this array in the innermost
        # loop of the Gibbs sampler. By forcing this array into 'F' order -
        # 'Fortran-contiguous' - we've set it so that accessing column slices
        # is faster. Tests indicate about 2X speed up in this operation from
        # 'F' order as opposed to the default 'C' order.
        self.known_source_cp = np.array(self.known_p_tv / self.denominator_p_v,
                                        order='F', dtype=np.float64)

        self.alpha2_n = self.alpha2 * self.n
        self.alpha2_n_tau = self.alpha2_n * self.tau

    def calculate_cp_slice(self, xi, m_xiV, m_V, n_vnoti):
        """Calculate slice of the conditional probability matrix.

        Parameters
        ----------
        xi : int
            Index of the column (taxon) of the conditional probability matrix
            that should be calculated.
        m_xiV : float
            Count of the training sequences (that are taxon xi) currently
            assigned to the unknown environment.
        m_V : float
            Sum of the training sequences currently assigned to the unknown
            environment (over all taxa).
        n_vnoti : float
            Counts of the test sequences in each environment at the current
            iteration of the sampler.

        Returns
        -------
        self.joint_probability : np.array
            The joint conditional probability distribution for the the current
            taxon based on the current state of the sampler.
        """
        # Components for known sources, i.e. indices {0,1...V-2}.
        self.joint_probability[:-1] = \
            self.known_source_cp[:, xi] * (n_vnoti[:-1] + self.beta)
        # Component for unknown source, i.e. index V-1.
        self.joint_probability[-1] = \
            ((m_xiV + self.alpha2_n) * (n_vnoti[-1] + self.beta)) / \
            ((m_V + self.alpha2_n_tau) * self.denominator_p_v)
        return self.joint_probability


def gibbs_sampler(sink, cp, restarts, draws_per_restart, burnin, delay):
    """Run Gibbs Sampler to estimate feature contributions from a sink sample.

    Parameters
    ----------
    sink : np.array
        A one dimentional array containing counts of features whose sources are
        to be estimated.
    cp : ConditionalProbability object
        Instantiation of the class handling probability calculations.
    restarts : int
        Number of independent Markov chains to grow. `draws_per_restart` *
        `restarts` gives the number of samplings of the mixing proportions that
        will be generated.
    draws_per_restart : int
        Number of times to sample the state of the Markov chain for each
        independent chain grown.
    burnin : int
        Number of passes (withdarawal and reassignment of every sequence in the
        sink) that will be made before a sample (draw) will be taken. Higher
        values allow more convergence towards the true distribtion before draws
        are taken.
    delay : int > 1
        Number passes between each sampling (draw) of the Markov chain. Once
        the burnin passes have been made, a sample will be taken every `delay`
        number of passes. This is also known as 'thinning'. Thinning helps
        reduce the impact of correlation between adjacent states of the Markov
        chain. Delay must be greater than 1, otherwise draws will never be
        taken. This is a legacy of the original R code.

    Returns
    -------
    final_envcounts : np.array
        2D array of ints. Rows are draws, columns are sources. The [i, j] entry
        is the number of sequences from draw i that where assigned to have come
        from environment j.
    final_env_assignments : np.array
        2D array of ints. Rows are draws, columns are conserved but arbitrary
        ordering. The [i, j] entry is the index of feature j in draw i. These
        orderings are identical for each draw.
    final_taxon_assignments : np.array
        2D array of ints. Rows are draws, columns are conserved but arbitrary
        ordering (same ordering as `final_env_assignments`). The [i, j] entry
        is the environment that the taxon `final_env_assignments[i, j]` is
        determined to have come from in draw i (j is the environment).
    """
    # Basic bookkeeping information we will use throughout the function.
    num_sources = cp.V
    num_features = cp.tau
    source_indices = np.arange(num_sources)
    sink = sink.astype(np.int32)
    sink_sum = sink.sum()

    # Calculate the number of passes that need to be conducted.
    total_draws = restarts * draws_per_restart
    total_passes = burnin + (draws_per_restart - 1) * delay + 1

    # Results containers.
    final_envcounts = np.zeros((total_draws, num_sources), dtype=np.int32)
    final_env_assignments = np.zeros((total_draws, sink_sum), dtype=np.int32)
    final_taxon_assignments = np.zeros((total_draws, sink_sum), dtype=np.int32)

    # Sequences from the sink will be randomly assigned a source environment
    # and then reassigned based on an increasingly accurate set of
    # probabilities. The order in which the sequences are selected for
    # reassignment must be random to avoid a systematic bias where the
    # sequences occuring later in the taxon_sequence book-keeping vector
    # receive more accurate reassignments by virtue of more updates to the
    # probability model. 'order' will be shuffled each pass, but can be
    # instantiated here to avoid unnecessary duplication.
    order = np.arange(sink_sum, dtype=np.int32)

    # Create a bookkeeping vector that keeps track of each sequence in the
    # sink. Each one will be randomly assigned an environment, and then
    # reassigned based on the increasinly accurate distribution. sink[i] i's
    # will be placed in the `taxon_sequence` vector to allow each individual
    # count to be removed and reassigned.
    taxon_sequence = np.repeat(np.arange(num_features), sink).astype(np.int32)

    # Update the conditional probability class now that we have the sink sum.
    cp.set_n(sink_sum)
    cp.precalculate()

    # Several bookkeeping variables that are used within the for loops.
    drawcount = 0
    unknown_idx = num_sources - 1

    for restart in range(restarts):
        # Generate random source assignments for each sequence in the sink
        # using a uniform distribution.
        seq_env_assignments, envcounts = \
            generate_environment_assignments(sink_sum, num_sources)

        # Initially, the count of each taxon in the 'unknown' source should be
        # 0.
        unknown_vector = np.zeros(num_features, dtype=np.int32)
        unknown_sum = 0

        # If a sequence's random environmental assignment is to the 'unknown'
        # environment we alter the training data to include those sequences
        # in the 'unknown' source.
        for e, t in zip(seq_env_assignments, taxon_sequence):
            if e == unknown_idx:
                unknown_vector[t] += 1
                unknown_sum += 1

        for rep in range(1, total_passes + 1):
            # Iterate through sequences in a random order so that no
            # systematic bias is introduced based on position in the taxon
            # vector (i.e. taxa appearing at the end of the vector getting
            # better estimates of the probability).
            np.random.shuffle(order)

            for seq_index in order:
                e = seq_env_assignments[seq_index]
                t = taxon_sequence[seq_index]

                # Remove the ith sequence and update the probability
                # associated with that environment.
                envcounts[e] -= 1
                if e == unknown_idx:
                    unknown_vector[t] -= 1
                    unknown_sum -= 1

                # Calculate the new joint probability vector based on the
                # removal of the ith sequence. Scale this probability vector
                # for use by np.random.choice.
                jp = cp.calculate_cp_slice(t, unknown_vector[t], unknown_sum,
                                           envcounts)

                # Reassign the sequence to a new source environment and
                # update counts for each environment and the unknown source
                # if necessary.
                new_e_idx = np.random.choice(source_indices, p=jp / jp.sum())

                seq_env_assignments[seq_index] = new_e_idx
                envcounts[new_e_idx] += 1

                if new_e_idx == unknown_idx:
                    unknown_vector[t] += 1
                    unknown_sum += 1

            if rep > burnin and ((rep-burnin) % delay) == 1:
                # Update envcounts array with the assigned envs.
                final_envcounts[drawcount] = envcounts

                # Assign vectors necessary for feature table reconstruction.
                final_env_assignments[drawcount] = seq_env_assignments
                final_taxon_assignments[drawcount] = taxon_sequence

                # We've made a draw, update this index so that the next
                # iteration will be placed in the correct index of results.
                drawcount += 1

    return (final_envcounts, final_env_assignments, final_taxon_assignments)


def _gibbs_loo(sources, alpha1, alpha2, beta, restarts, draws_per_restart,
               burnin, delay, cluster=None, create_feature_tables=True):
    '''Gibb's LOO sampling API - see _gibbs for documentation.

    Notes
    -----
    In leave-one-out (LOO) classification, each source is individually
    considered as a sink, so there is no need to pass a sinks dataframe.
    '''
    def f(cp_and_sink):
        # The import is here to ensure that the engines of the cluster can
        # access the gibbs_sampler function.
        from sourcetracker._sourcetracker import gibbs_sampler
        return gibbs_sampler(cp_and_sink[1], cp_and_sink[0], restarts,
                             draws_per_restart, burnin, delay)
    cps_and_sinks = []
    for source in sources.index:
        _sources = sources.select(lambda x: x != source)
        cp = ConditionalProbability(alpha1, alpha2, beta, _sources.values)
        sink = sources.loc[source, :].values
        cps_and_sinks.append((cp, sink))

    if cluster is not None:
        results = cluster[:].map(f, cps_and_sinks, block=True)
    else:
        results = list(map(f, cps_and_sinks))
    mpm, mps, fas = collate_gibbs_results([i[0] for i in results],
                                          [i[1] for i in results],
                                          [i[2] for i in results],
                                          sources.index, sources.index,
                                          sources.columns,
                                          create_feature_tables, loo=True)
    return mpm, mps, fas


def _gibbs(sources, sinks, alpha1, alpha2, beta, restarts, draws_per_restart,
           burnin, delay, cluster=None, create_feature_tables=True):
    '''Gibb's sampling API.

    Notes
    -----
    This function exists to allow API calls to source/sink prediction. It is a
    candidate public API call. You can track progress on this via
    https://github.com/biota/sourcetracker2/issues/31

    Warnings
    --------
    This function does _not_ perform rarefaction, the user should perform
    rarefaction prior to calling this function. This function also does not
    perform checks on the data (i.e. using `check_and_correct_data`, or
    `intersect_and_sort_samples`). Finally, this function does not collapse
    sources or sinks, it expects each row of the `sources` dataframe to
    represent a unique source, and each row of the `sinks` dataframe to
    represent a unique sink.

    Parameters
    ----------
    sources : DataFrame
        A dataframe containing source data (rows are sources, columns are
        features). The index must be the names of the sources.
    sinks : DataFrame
        A dataframe containing sink data (rows are sinks, columns are
        features). The index must be the names of the sinks.
    alpha1 : float
        Prior counts of each feature in the training environments. Higher
        values decrease the trust in the training environments, and make
        the source environment distributions over taxa smoother. By
        default, this is set to 0.001, which indicates reasonably high
        trust in all source environments, even those with few training
        sequences. This is useful when only a small number of biological
        samples are available from a source environment. A more
        conservative value would be 0.01.
    alpha2 : float
        Prior counts of each feature in the Unknown environment. Higher
        values make the Unknown environment smoother and less prone to
        overfitting given a training sample.
    beta : float
        Number of prior counts of test sequences from each feature in each
        environment.
    restarts : int
        Number of independent Markov chains to grow. `draws_per_restart` *
        `restarts` gives the number of samplings of the mixing proportions that
        will be generated.
    draws_per_restart : int
        Number of times to sample the state of the Markov chain for each
        independent chain grown.
    burnin : int
        Number of passes (withdarawal and reassignment of every sequence in the
        sink) that will be made before a sample (draw) will be taken. Higher
        values allow more convergence towards the true distribtion before draws
        are taken.
    delay : int > 1
        Number passes between each sampling (draw) of the Markov chain. Once
        the burnin passes have been made, a sample will be taken every `delay`
        number of passes. This is also known as 'thinning'. Thinning helps
        reduce the impact of correlation between adjacent states of the Markov
        chain. Delay must be greater than 1, otherwise draws will never be
        taken. This is a legacy of the original R code.
    cluster : ipyparallel.client.client.Client or None
        An ipyparallel Client object, e.g. a started cluster.
    create_feature_tables : boolean
        If `True` create a feature table for each sink. The feature table
        records the average count of each feature from each source for this
        sink. This option can consume large amounts of memory if there are many
        source, sinks, and features. If `False`, feature tables are not
        created.

    Returns
    -------
    mpm : DataFrame
        Mixing proportion means. A dataframe containing the mixing proportions
        (rows are sinks, columns are sources).
    mps : DataFrame
        Mixing proportion standard deviations. A dataframe containing the
        mixing proportions standard deviations (rows are sinks, columns are
        sources).
    fas : list
        ith item is a pd.DataFrame of the average feature assignments from each
        source for the ith sink (in the same order as rows of `mpm` and `mps`).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ipyparallel import Client
    >>> import subprocess
    >>> import time
    >>> from sourcetracker._sourcetracker import _gibbs

    # Prepare some source data.
    >>> otus = np.array(['o%s' % i for i in range(50)])
    >>> source1 = np.random.randint(0, 1000, size=50)
    >>> source2 = np.random.randint(0, 1000, size=50)
    >>> source3 = np.random.randint(0, 1000, size=50)
    >>> source_df = pd.DataFrame([source1, source2, source3],
                                 index=['source1', 'source2', 'source3'],
                                 columns=otus, dtype=np.float64)

    # Prepare some sink data.
    >>> sink1 = np.ceil(.5*source1+.5*source2)
    >>> sink2 = np.ceil(.5*source2+.5*source3)
    >>> sink3 = np.ceil(.5*source1+.5*source3)
    >>> sink4 = source1
    >>> sink5 = source2
    >>> sink6 = np.random.randint(0, 1000, size=50)
    >>> sink_df = pd.DataFrame([sink1, sink2, sink3, sink4, sink5, sink6],
                               index=np.array(['sink%s' % i for i in
                                               range(1,7)]),
                               columns=otus, dtype=np.float64)

    # Set paramaters
    >>> alpha1 = .01
    >>> alpha2 = .001
    >>> beta = 10
    >>> restarts = 5
    >>> draws_per_restart = 1
    >>> burnin = 2
    >>> delay = 2

    # Call without a cluster
    >>> mpm, mps, fas = _gibbs(source_df, sink_df, alpha1, alpha2, beta,
                               restarts, draws_per_restart, burnin, delay,
                               cluster=None, create_feature_tables=True)

    # Start a cluster and call the function
    >>> jobs = 4
    >>> subprocess.Popen('ipcluster start -n %s --quiet' % jobs, shell=True)
    >>> time.sleep(25)
    >>> c = Client()
    >>> mpm, mps, fas = _gibbs(source_df, sink_df, alpha1, alpha2, beta,
                               restarts, draws_per_restart, burnin, delay,
                               cluster=c, create_feature_tables=True)
    '''
    cp = ConditionalProbability(alpha1, alpha2, beta, sources.values)
    f = partial(gibbs_sampler, cp=cp, restarts=restarts,
                draws_per_restart=draws_per_restart, burnin=burnin,
                delay=delay)
    if cluster is not None:
        results = cluster[:].map(f, sinks.values, block=True)
    else:
        results = list(map(f, sinks.values))
    mpm, mps, fas = collate_gibbs_results([i[0] for i in results],
                                          [i[1] for i in results],
                                          [i[2] for i in results],
                                          sinks.index, sources.index,
                                          sources.columns,
                                          create_feature_tables, loo=False)
    return mpm, mps, fas


def cumulative_proportions(all_envcounts, sink_ids, source_ids):
    '''Calculate contributions of each source for each sink in `sink_ids`.

    Parameters
    ----------
    all_envcounts : list
        Each entry is 2D array of ints. The ith entry must correspond to the
        ith sink ID. The [j, k] entry of the ith table is the count of
        sequences assigned to the sink from kth environment during the jth
        draw.
    sink_ids : np.array
        ID's of the sinks. Must be in the same order as data in
        `all_envcounts`.
    source_ids : np.array
        ID's of the sources. Must be in the same order as the columns of the
        tables in `all_envcounts`.

    Returns
    -------
    proportions : pd.DataFrame
        A dataframe of floats, containing the mixing proportions of each source
        in each sink. The [i, j] entry is the contribution from the jth source
        to the ith sink.
    proportions_std : pd.DataFrame
        A dataframe of floats, identical to `proportions` except the entries
        are the standard deviation of each entry in `proportions`.

    Notes
    -----
    This script is designed to be used by `collate_gibbs_results` after
    completion of multiple `gibbs_sampler` calls (for different sinks). This
    function does _not_ check that the assumptions of ordering described above
    are met. It is the user's responsibility to check these if using this
    function independently.
    '''
    num_sinks = len(sink_ids)
    num_sources = len(source_ids) + 1

    proportions = np.zeros((num_sinks, num_sources), dtype=np.float64)
    proportions_std = np.zeros((num_sinks, num_sources), dtype=np.float64)

    for i, envcounts in enumerate(all_envcounts):
        proportions[i] = envcounts.sum(0) / envcounts.sum()
        proportions_std[i] = (envcounts / envcounts.sum()).std(0)

    cols = list(source_ids) + ['Unknown']
    return (pd.DataFrame(proportions, index=sink_ids, columns=cols),
            pd.DataFrame(proportions_std, index=sink_ids, columns=cols))


def single_sink_feature_table(final_env_assignments, final_taxon_assignments,
                              source_ids, feature_ids):
    '''Produce a feature table from the output of `gibbs_sampler`.

    Parameters
    ----------
    final_env_assignments : np.array
        2D array of ints. The [i, j] entry is the environment that sequence j
        was assigned in the ith draw. The ordering of the columns is determined
        by `np.repeat` and the count of different features in the sink.
        The shape is number of draws by sum of the sink.
    final_taxon_assignments : np.array
        2D array of ints. The [i, j] entry is the index of feature j in all
        features, that was assigned in the ith draw. The ordering of the
        columns is determined by `np.repeat` and the count of different
        features in the sink. The shape is number of draws by sum of the sink.
    source_ids : np.array
        ID's of the sources.
    feature_ids : np.array
        ID's of the features.

    Returns
    -------
    pd.DataFrame
        A dataframe containing counts of features contributed to the sink by
        each source.

    Notes
    -----
    This script is designed to be used by `collate_gibbs_results` after
    completion of multiple `gibbs_sampler` calls (for different sinks). This
    function does _not_ check that the assumptions of ordering described above
    are met. It is the user's responsibility to check these if using this
    function independently.
    '''
    num_sources = len(source_ids) + 1
    num_features = len(feature_ids)
    data = np.zeros((num_sources, num_features), dtype=np.int32)
    for r, c in zip(final_env_assignments.ravel(),
                    final_taxon_assignments.ravel()):
        data[r, c] += 1
    return pd.DataFrame(data, index=list(source_ids) + ['Unknown'],
                        columns=feature_ids)


def collate_gibbs_results(all_envcounts, all_env_assignments,
                          all_taxon_assignments, sink_ids, source_ids,
                          feature_ids, create_feature_tables, loo):
    '''Collate `gibbs_sampler` output, optionally including feature tables.

    Parameters
    ----------
    all_envcounts : list
        Each entry is 2D array of ints. The ith entry must correspond to the
        ith sink ID. The [j, k] entry of the ith table is the count of
        sequences assigned to the sink from kth environment during the jth
        draw.
    all_env_assignments : list
        Each entry is a 2D array of ints. The ith entry is the environment
        assignments for the ith sink. The [j, k] cell of the ith entry is the
        environment of the kth taxon from the jth draw.
    all_taxon_assignments : list
        Each entry is a 2D array of ints. The ith entry is the feature indices
        (over all features) for the ith sink. The [j, k] cell of the ith entry
        is the feature index of the kth taxon selected for removal and
        reassignment in the jth draw.
    sink_ids : np.array
        ID's of the sinks.
    source_ids : np.array
        ID's of the sources.
    feature_ids : np.array
        ID's of the features.
    create_feature_tables : boolean
        If `True` create a feature table for each sink. The feature table
        records the average count of each feature from each source for this
        sink. This option can consume large amounts of memory if there are many
        source, sinks, and features. If `False`, feature tables are not
        created.
    loo : boolean
        If `True`, collate data based on the assumption that input data was
        generated by a `gibbs_loo` call.

    Notes
    -----
    This script is designed to be used by after completion of multiple
    `gibbs_sampler` calls (for different sinks). This function does _not_ check
    that the assumptions of ordering described below are met. It is the user's
    responsibility to check these if using this function independently.

    If `loo=False`, the order of the entries in each list (first 3 inputs) must
    be the same, and correspond to the order of the `sink_ids`.

    If `loo=True`, the order of the entries in each list (first 3 inputs) must
    be the same, and correspond to the order of the `source_ids`.
    '''
    if loo:
        props, props_stds = cumulative_proportions(all_envcounts, source_ids,
                                                   source_ids[:-1])
        # The source_ids for each environment are different. Specifically, the
        # ith row of `props` and `props_stds` must have a 0 value inserted at
        # the ith position to reflect the fact that the ith source was held out
        # (it was the sink during that iteration). To do this we can imagine
        # breaking the nXn array returned by `cumulative_proportions`, and
        # inserting it into an nXn+1 array, with the missing cells on the
        # diagonal of the nXn+1 array.
        nrows = len(source_ids)
        ncols = nrows + 1
        new_source_ids = list(source_ids)+['Unknown']
        new_data = np.zeros((nrows, ncols), dtype=np.float64)
        new_data_std = np.zeros((nrows, ncols), dtype=np.float64)

        new_data[np.triu_indices(ncols, 1)] = \
            props.values[np.triu_indices(ncols - 1, 0)]
        new_data[np.tril_indices(ncols - 1, -1, ncols)] = \
            props.values[np.tril_indices(ncols - 1, -1)]

        new_data_std[np.triu_indices(ncols, 1)] = \
            props_stds.values[np.triu_indices(ncols - 1, 0)]
        new_data_std[np.tril_indices(ncols - 1, -1, ncols)] = \
            props_stds.values[np.tril_indices(ncols - 1, -1)]

        props = pd.DataFrame(new_data, index=source_ids,
                             columns=new_source_ids)
        props_stds = pd.DataFrame(new_data_std, index=source_ids,
                                  columns=new_source_ids)

        if create_feature_tables:
            fts = []
            for i, sink_id in enumerate(source_ids):
                r_source_ids = source_ids[source_ids != sink_id]
                ft = single_sink_feature_table(all_env_assignments[i],
                                               all_taxon_assignments[i],
                                               r_source_ids, feature_ids)
                tmp = ft.T
                tmp.insert(i, sink_id, 0)
                fts.append(tmp.T)
        else:
            fts = None

    # LOO not done.
    else:
        props, props_stds = cumulative_proportions(all_envcounts, sink_ids,
                                                   source_ids)
        if create_feature_tables:
            fts = []
            for i, sink_id in enumerate(sink_ids):
                ft = single_sink_feature_table(all_env_assignments[i],
                                               all_taxon_assignments[i],
                                               source_ids, feature_ids)
                fts.append(ft)
        else:
            fts = None

    return props, props_stds, fts