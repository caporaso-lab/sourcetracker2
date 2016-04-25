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
from copy import copy
import numpy as np
from skbio.stats._subsample import subsample_counts
import pandas as pd
from functools import partial
from biom.table import Table
import glob
from tempfile import TemporaryDirectory


def parse_mapping_file(mf_lines):
    """Parse lines from a mapping file into a nested dictionary.

    This function copies qiime.parse_mapping_file_to_dict, but does _not_ allow
    commented lines at the top of a mapping file.

    Parameters
    ----------
    mf_lines : list
        Each entry of the list is a tab delimited string corresponding to a row
        of the mapping file.

    Returns
    -------
    results : dict
    """
    headers = mf_lines[0].strip().split('\t')[1:]
    results = {}
    for line in mf_lines[1:]:
        tmp = line.strip().split('\t')
        results[tmp[0]] = {k: v.strip() for k, v in zip(headers, tmp[1:])}
    return results


def collapse_sources(samples, sample_metadata, category, biom_table,
                     sort=True):
    """Collapse sources based on metadata under `category`.

    Parameters
    ----------
    samples : list
        Sample id's that contain data to be summed and collapsed.
    sample_metadata : dict
        A mapping file that has been parsed with parse_mapping_file. The
        keys are sample ids, and the values are dicts with each header in the
        mapping file as a key and the values from that sample under that
        header.
    category : str
        Header in the mapping file (and thus each second level dictionary of
        `sample_metadata`) that identifies the source environment of each
        sample.
    biom_table : biom.table.Table
        A biom table containing source data.
    sort : boolean, optional
        If true, the set of sources will be sorted using np.argsort.

    Returns
    -------
    envs : np.array
        Source environments in the same order as collapsed_sources.
    collapsed_sources: np.array
        Columns are features (OTUs), rows are collapsed samples. The [i,j]
        entry is the sum of the counts of features j in all samples which were
        considered part of source i.

    Notes
    -----
    `biom_table` will have S samples = [s0, s1, ... sn]. Each of these samples
    has a metadata value under `category` = [c0, c1, ... cn]. This function
    will produce an array with K rows, where the ith row has the data from the
    ith member of the set of category values. As an example:

    Input table data:
        s1  s2  s3  s4
    o1  10  50  10  70
    o2  0   25  10  5
    o3  0   25  10  5
    o4  100 0   10  5

    mf:
        cat1
    s1  A
    s2  A
    s3  A
    s4  B

    result of using 'cat1':

    Collapsed table data:
            o1  o2  o3  o4
    A       70  35  35  110
    B       70  5   5   5
    """
    envs = []
    sample_groups = []
    for s in samples:
        env = sample_metadata[s][category]
        try:
            idx = envs.index(env)
            sample_groups[idx].append(s)
        except ValueError:
            envs.append(env)
            idx = envs.index(env)
            sample_groups.append([])
            sample_groups[idx].append(s)

    collapsed_sources = np.zeros((len(envs), biom_table.shape[0]), dtype=int)
    for row, group in enumerate(sample_groups):
        for sample in group:
            collapsed_sources[row] += biom_table.data(sample, axis='sample',
                                                      dense=True).astype(int)
    if sort:
        indices = np.argsort(envs)
        return np.array(envs)[indices], collapsed_sources[indices]
    else:
        return np.array(envs), collapsed_sources


def subsample_sources_sinks(sources_data, sinks, feature_table, sources_depth,
                            sinks_depth):
    '''Rarify data for sources and sinks.

    Notes
    -----
    This function rarifies `sources_data` to `sources_depth`, and `sinks` in
    `feature_table` to `sink_depth`. This function is neccesary because of
    ipyparallel and the partial functions.

    Parameters
    ----------
    sources_data : np.array
        Two dimensional array with collapsed source data.
    sinks : np.array
        One dimensional array of strings, with each string being the sample ID
        of a sink in `feature_table`.
    feature_table : biom.table.Table
        Biom table containing data for `sinks` to be rarified.
    sources_depth : int
        Depth at which to subsample each source. If 0, no rarefaction will be
        performed.
    sinks_depth : int
        Depth at which to subsample each sink. If 0, no rarefaction will be
        performed.

    Returns
    -------
    rsd : np.array
        Rarified `sources_data`.
    rft : biom.table.Table
        `feature_table` with samples identified in `sinks` rarified.
    '''
    # Check that supplied depths do not exceed available sequences. Cryptic
    # errors will be raised otherwise.
    if sources_depth > 0 and (sources_data.sum(1) < sources_depth).any():
        raise ValueError('Invalid rarefaction depth for source data. There '
                         'are not enough sequences in at least one collapsed '
                         'source.')
    if sinks_depth > 0:
        for sample in sinks:
            if feature_table.data(sample, axis='sample').sum() < sinks_depth:
                raise ValueError('Invalid rarefaction depth for sink data. '
                                 'There are not enough sequences in at least '
                                 'one sink.')

    # Rarify source data.
    if sources_depth == 0:
        rsd = sources_data
    else:
        rsd = np.empty(sources_data.shape, dtype=np.float64)
        for row in range(sources_data.shape[0]):
            rsd[row] = subsample_counts(sources_data[row], sources_depth,
                                        replace=False)
    # Rarify sinks data in the biom table.
    if sinks_depth == 0:
        rft = feature_table
    else:
        # We'd like to use Table.subsample, but it removes features that have
        # 0 count across every sample, which changes the size of the matrix.
        # rft = feature_table.filter(sinks, axis='sample', inplace=False)
        # rft = rft.subsample(sinks_depth)
        def _rfx(data, sid, md):
            if sid in sinks:
                return subsample_counts(data.astype(np.int64), sinks_depth,
                                        replace=False)
            else:
                return data
        rft = feature_table.transform(_rfx, axis='sample', inplace=False)
    return rsd, rft


class Sampler(object):
    def __init__(self, sink_data, num_sources):
        """Initialize Sampler.

        Parameters
        ----------
        sink_data : np.array
            Counts of each OTU in given sink.
        num_sources : int
            Number of sources, including unknown.

        Attributes
        ----------
        self.sink_data : np.array
            Sink data (see Parameters).
        self.num_sources : int
            Number of source environments.
        self.num_features : int
            Number of features.
        self.sum : int
            Total number of sequences in sink sample.
        self.taxon_sequence : np.array
            Bookkeeping array that contains an entry for each sequence so that
            their source assignments can be manipulated. Not set until
            `generate_taxon_sequence` is called.
        self.seq_env_assignments : np.array
            Vector of integers where the ith entry is the source environment
            index of the ith sequence. Not set until
            `generate_environment_assignments` is called.
        self.envcounts : np.array
            Total number of sequences assigned to each source environment. Not
            set until `generate_environment_assignments` is called.
        """
        self.sink_data = sink_data.astype(np.int64)
        self.sum = self.sink_data.sum()
        self.num_sources = num_sources
        self.num_features = sink_data.shape[0]

    def generate_taxon_sequence(self):
        """Generate vector of taxa containing each sequence in self.sink_data.

        Notes
        -----
        This function is used for book-keeping. The ith value of taxon_sequence
        is the (integer) index of an OTU in self.sink_data. As an example, if
        the sink data were [2, 0, 0, 4] (OTU_0 has an abundance of 2 and OTU_3
        has an abundance of 4), the taxon sequence would be [0, 0, 3, 3, 3, 3].
        """
        self.taxon_sequence = np.empty(self.sum, dtype=np.int32)
        k = 0
        for ind, num in enumerate(self.sink_data):
            self.taxon_sequence[k: k + num] = ind
            k += num

    def generate_environment_assignments(self):
        """Randomly assign each sequence in the sink to a source environment.

        Notes
        -----
        Assignments to a source are made from a uniform distribtuion.

        """
        choices = np.arange(self.num_sources)
        self.seq_env_assignments = np.random.choice(choices, size=self.sum,
                                                    replace=True)
        self.envcounts = np.bincount(self.seq_env_assignments,
                                     minlength=self.num_sources)

    def seq_assignments_to_contingency_table(self):
        """Return contingency table built from `self.taxon_sequence`.

        Returns
        -------
        ct : np.array
            Two dimensional array with rows as features, sources as columns.
            The [i, j] entry is the count of feature i assigned to source j.
        """
        ct = np.zeros((self.num_features, self.num_sources))
        for i in range(int(self.sum)):
            ct[self.taxon_sequence[i], self.seq_env_assignments[i]] += 1
        return ct


class ConditionalProbability(object):
    def __init__(self, alpha1, alpha2, beta, source_data):
        r"""Set properties used for calculating the conditional probability.

        Paramaters
        ----------
        alpha1 : float
            Prior counts of each species in the training environments. Higher
            values decrease the trust in the training environments, and make
            the source environment distributions over taxa smoother. By
            default, this is set to 0.001, which indicates reasonably high
            trust in all source environments, even those with few training
            sequences. This is useful when only a small number of biological
            samples are available from a source environment. A more
            conservative value would be 0.01.
        alpha2 : float
            Prior counts of each species in the Unknown environment. Higher
            values make the Unknown environment smoother and less prone to
            overfitting given a training sample.
        beta : float
            Number of prior counts of test sequences from each OTU in each
            environment
        source_data : np.array
            Columns are features (OTUs), rows are collapsed samples. The [i,j]
            entry is the sum of the counts of features j in all samples which
            were considered part of source i.

        Attributes
        ----------
        self.m_xivs : np.array
            This is an exact copy of the source_data passed when the function
            is initialized. It is referenced as m_xivs because m_xiv is the
            [v, xi] entry of the source data. In other words, the count of the
            xith feature in the vth environment.
        self.m_vs : np.array
            The row sums of self.m_xivs. This is referenced as m_v in [1]_.
        self.V : int
            Number of environments (includes both known sources and the
            'unknown' source).
        self.tau : int
            Number of features.
        self.joint_probability : np.array
            The joint conditional distribution. Until the `precalculate` method
            is called, this will be uniformly zero.
        self.n : int
            Number of sequences in the sink.
        self.known_p_tv : np.array
            An array giving the precomputable parts of the probability of
            finding the xith taxon in the vth environment given the known
            sources, aka p_tv in the R implementation. Rows are (known)
            sources, columns are features, shape is (V-1, tau).
        self.denominator_p_v : float
            The denominator of the calculation for finding the probability of
            a sequence being in the vth environment given the training data
            (source data).
        self.known_source_cp : np.array
            All precomputable portions of the conditional probability array.
            Dimensions are the same as self.known_p_tv.

        Notes
        -----
        This class exists to calculate the conditional probability given in
        reference [1]_ (with modifications based on communications with the
        author). Since the calculation of the conditional probability must
        occur during each pass of the Gibbs sampler, reducing the number of
        computations is of paramount concern. This class precomputes everything
        that is static throughout a run of the sampler to reduce the innermost
        for loop computations.

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
        and then iteratively reassigned) and which taxon (the slice) the
        sequence we have removed was from.
        >>> xi = 2
        Count of the training sequences (that are taxon xi) currently assigned
        to the unkown environment.
        >>> m_xiV = 38
        Sum of the training sequences currently assigned to the unkown
        environment (over all taxa).
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
        self.m_xivs = source_data
        self.m_vs = np.expand_dims(source_data.sum(1), axis=1)
        self.V = source_data.shape[0] + 1
        self.tau = source_data.shape[1]
        # Create the joint probability vector which will be overwritten each
        # time self.calculate_cp_slice is called.
        self.joint_probability = np.zeros(self.V)

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
        # 'Fortran-contiguous' - we've set it so that acessing column slices is
        # faster. Tests indicate about 2X speed up in this operation from 'F'
        # order as opposed to the default 'C' order.
        self.known_source_cp = np.array(self.known_p_tv / self.denominator_p_v,
                                        order='F')

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
            assigned to the unkown environment.
        m_V : float
            Sum of the training sequences currently assigned to the unkown
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


def gibbs_sampler(cp, sink, restarts, draws_per_restart, burnin, delay):
    """Run Gibbs Sampler to estimate feature contributions from a sink sample.

    Parameters
    ----------
    cp : ConditionalProbability object
        Instantiation of the class handling probability calculations.
    sink : np.array
        A one dimentional array containing counts of features whose sources are
        to be estimated.
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
    mixing_proportions : np.array
        A two dimensional array containing the estimated source proportions.
        The [i,j] entry is the estimated proportion of the sink sample coming
        from source environment j calculated during draw i.
    calculated_assignments : np.array
        A three dimensional array containing the source environment assignments
        for each sequence in the sample. The first dimension of the array is
        the 'draw'; this is analagous to the first dimension in
        sink_proportions. The second and third dimensions are features and
        sources, respectively. Thus the [i,j,k] entry is the number of
        sequences from the sink sample that were taxon j, from environment k,
        in the ith independent draw.
    """
    # Basic bookkeeping information we will use throughout the function.
    num_sources = cp.V
    num_features = cp.tau
    source_indices = np.arange(num_sources)

    # Calculate the number of passes that need to be conducted.
    total_draws = restarts * draws_per_restart
    total_passes = burnin + (draws_per_restart - 1) * delay + 1

    # Results containers.
    mixing_proportions = np.empty((total_draws, num_sources))
    calculated_assignments = np.empty((total_draws, num_features, num_sources))

    # Sequences from the sink will be randomly assigned a source environment
    # and then reassigned based on an increasingly accurate set of
    # probabilities. The order in which the sequences are selected for
    # reassignment must be random to avoid a systematic bias where the
    # sequences occuring later in the taxon_sequence book-keeping vector
    # receive more accurate reassignments by virtue of more updates to the
    # probability model. 'order' will be shuffled each pass, but can be
    # instantiated here to avoid unnecessary duplication.
    sink_sum = sink.sum()
    order = np.arange(sink_sum, dtype=np.int)

    # Create a bookkeeping vector that keeps track of each sequence in the
    # sink. Each one will be randomly assigned an environment, and then
    # reassigned based on the increasinly accurate distribution.
    sampler = Sampler(sink, num_sources)
    sampler.generate_taxon_sequence()

    # Update the conditional probability class now that we have the sink sum.
    cp.set_n(sink_sum)
    cp.precalculate()

    # Several bookkeeping variables that are used within the for loops.
    drawcount = 0
    unknown_idx = num_sources - 1

    for restart in range(restarts):
        # Generate random source assignments for each sequence in the sink
        # using a uniform distribution.
        sampler.generate_environment_assignments()

        # Initially, the count of each taxon in the 'unknown' source should be
        # 0.
        unknown_vector = np.zeros(num_features)
        unknown_sum = 0

        # If a sequence's random environmental assignment is to the 'unknown'
        # environment we alter the training data to include those sequences
        # in the 'unknown' source.
        for e, t in zip(sampler.seq_env_assignments, sampler.taxon_sequence):
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
                e = sampler.seq_env_assignments[seq_index]
                t = sampler.taxon_sequence[seq_index]

                # Remove the ith sequence and update the probability
                # associated with that environment.
                sampler.envcounts[e] -= 1
                if e == unknown_idx:
                    unknown_vector[t] -= 1
                    unknown_sum -= 1

                # Calculate the new joint probability vector based on the
                # removal of the ith sequence. Scale this probability vector
                # for use by np.random.choice.
                jp = cp.calculate_cp_slice(t, unknown_vector[t], unknown_sum,
                                           sampler.envcounts)

                # Reassign the sequence to a new source environment and
                # update counts for each environment and the unknown source
                # if necessary.
                jp_sum = jp.sum()
                new_e_idx = np.random.choice(source_indices, p=jp / jp_sum)

                sampler.seq_env_assignments[seq_index] = new_e_idx
                sampler.envcounts[new_e_idx] += 1
                if new_e_idx == unknown_idx:
                    unknown_vector[t] += 1
                    unknown_sum += 1

            if rep > burnin and ((rep-burnin) % delay) == 1:
                # Calculate proportion of sample from each source and record
                # this as the source proportions for this sink.
                prps = sampler.envcounts / sink_sum
                prps_sum = prps.sum()
                mixing_proportions[drawcount] = prps/prps_sum

                # Each sequence in the sink sample has been assigned to a
                # source environment. Convert this information into a
                # contingency table.
                calculated_assignments[drawcount] = \
                    sampler.seq_assignments_to_contingency_table()

                # We've made a draw, update this index so that the next
                # iteration will be placed in the correct slice of the
                # mixing_proportions and calculated_assignments arrays.
                drawcount += 1

    return mixing_proportions, calculated_assignments


def sinks_and_sources(sample_metadata):
    '''Return lists of source and sink samples.

    Notes
    -----
    This function assumes that source and sink samples will be delineated by
    having the value `source` or `sink` under the header `SourceSink` in
    `sample_metadata`.

    Parameters
    ----------
    sample_metadata : dict
        Dictionary containing sample metadata. Format is common QIIME format.

    Returns
    -------
    source_samples : list
        Samples that will be collapsed and used as 'Sources'.
    sink_samples : list
        Samples that will be used as 'Sinks'.
    '''
    sink_samples = []
    source_samples = []
    for sample, md in sample_metadata.items():
        if md['SourceSink'] == 'sink':
            sink_samples.append(sample)
        elif md['SourceSink'] == 'source':
            source_samples.append(sample)
        else:
            pass
    return source_samples, sink_samples


def _cli_sync_biom_and_sample_metadata(sample_metadata, biom_table):
    """Reduce mapping file dict and biom table to shared samples.

    Notes
    -----
    This function is used by the command line interface (CLI) to allow mapping
    files and biom tables which have disjoint sets of samples to be used in the
    script without forcing the user to keep multiple copies of either.

    Parameters
    ----------
    sample_metadata : dict
        Traditional parsed QIIME mapping file, i.e. nested dictionary with
        sample ids as keys to a dictionary with mapping file headers as keys
        and value for that sample the entries under those headers for that
        sample.
    biom_table : biom.Table.table
        Biom table.

    Returns
    -------
    nbiom_table : biom.Table.table
        Biom table with shared samples only.
    nsample_metadata : dict
        Sample metadata dictionary with only shared samples.

    Raises
    ------
    ValueError
            If there are no shared samples a ValueError is raised.
    """
    mf_samples = set(sample_metadata)
    biom_table_samples = set(biom_table.ids(axis='sample'))
    if mf_samples == biom_table_samples:
        return sample_metadata, biom_table
    else:
        shared_samples = mf_samples.intersection(biom_table_samples)
        if len(shared_samples) == 0:
            err_text = ('No samples were shared between sample metadata and '
                        'biom file.')
            raise ValueError(err_text)
        # Remove samples that were in the mapping file but not biom file.
        nsample_metadata = {k: v for k, v in sample_metadata.items() if k in
                            shared_samples}

        def _f(sv, sid, smd):
            '''Remove samples in biom table that are not in mapping file.'''
            return sid in shared_samples
        nbiom_table = biom_table.filter(_f, axis='sample', inplace=False)
    return nsample_metadata, nbiom_table


def _cli_collate_results(samples, sample_data, env_ids):
    """Collate results from individual samples.

    Notes
    -----
    This function is used by the command line interface (CLI) to allow
    individually written samples to be collated into two results arrays.

    Parameters
    ----------
    samples : list
        Names of the samples (sinks) in the same order as their data in
        `sample_data`.
    sample_data : list
        Arrays of sample data, where each array contains the contributions from
        each source for each draw. In the same order as `samples`.
    env_ids : np.array
        An containing the source environment identifiers, generated from
        `collapse_to_known_sources`.

    Returns
    -------
    mean_lines : str
        String representing contingency table containing mean contribution from
        each environment for each sink.
    std_lines : str
        String representing contingency table containing standard deviation of
        contribution from each environment for each sink.
    """
    mixing_proportions = []
    mixing_proportions_stds = []
    for sample, data in zip(samples, sample_data):
        mixing_proportions.append(data.mean(axis=0))
        mixing_proportions_stds.append(data.std(axis=0))

    header = '\t'.join(['SampleID'] + env_ids.tolist() + ['Unknown'])
    mean_lines = [header]
    std_lines = [header]

    for sample_id, means, stds in zip(samples, mixing_proportions,
                                      mixing_proportions_stds):
        mean_lines.append('\t'.join([sample_id] + list(map(str, means))))
        std_lines.append('\t'.join([sample_id] + list(map(str, stds))))

    return '\n'.join(mean_lines), '\n'.join(std_lines)


def _cli_single_sample_formatter(proportions):
    """Prepare data from a Gibb's run on a single sample for writing.

    Notes
    -----
    This function is used by the CLI to prepare for writing the results of the
    Gibb's sampler on a single sink sample to disk. This function serves two
    purposes:
    1) By writing the results of individual samples to disk, a failed
       or interrupted run has all but the sample being currently worked on
       saved.
    2) Computation for individual samples can be split amongst multiple
       processors because each will operate only on a single sample and write
       that result to disk.

    Parameters
    ----------
    proportions : np.array
        Two dimensional array containing calculated source proportions
        (columns) by independent draws (rows).

    Returns
    -------
    lines : str
        String ready to be written containing the `mixing_proportions` data.
    """
    return '\n'.join(['\t'.join(list(map(str, row))) for row in proportions])


def _cli_sink_source_prediction_runner(sample, alpha1, alpha2, beta, restarts,
                                       draws_per_restart, burnin, delay,
                                       sources_data, biom_table, output_dir):
    """Run sink source prediction.

    Notes
    -----
    This function is used by the CLI to allow ipyparallels to map different
    samples to different processors and enable parallelization. The parameters
    passed to this function not described below are described in the
    ConditionalProbability class, the Sampler class, or the gibbs function.

    Parameters
    ----------
    sample : str
        ID for the sample whose sources should be predicted.
    sources_data : np.array
        Data detailing the source environments.
    output_dir : str
        Path to the output directory where the results will be saved.
    """
    cp = ConditionalProbability(alpha1, alpha2, beta, sources_data)
    if isinstance(biom_table, pd.core.frame.DataFrame):
        sink_data = biom_table.loc[sample].values
    elif isinstance(biom_table, Table):
        sink_data = biom_table.data(sample, axis='sample', dense=True)
    else:
        raise TypeError('OTU table data is neither Pandas DataFrame nor '
                        'biom.table.Table. These are the only supported '
                        'formats.')

    results = gibbs_sampler(cp, sink_data, restarts, draws_per_restart, burnin,
                            delay)
    lines = _cli_single_sample_formatter(results[0])
    o = open(os.path.join(output_dir, sample + '.txt'), 'w')
    o.writelines(lines)
    o.close()
    # calculated_assignments.append(results[1])


def _cli_loo_runner(sample, source_category, alpha1, alpha2, beta, restarts,
                    draws_per_restart, burnin, delay, sample_metadata,
                    sources_data, sources_envs, biom_table, output_dir):
    """Run leave-one-out source contribution prediction.

    Notes
    -----
    This function is used by the CLI to allow ipyparallels to map different
    samples to different processors and enable parallelization. The parameters
    passed to this function not described below are described in the
    ConditionalProbability class, the Sampler class, or the gibbs function.

    Parameters
    ----------
    sample : str
        ID for the sample whose sources should be predicted.
    source_category : str
        Key for `sample_metadata[sample]` that indicates which source the
        sample belongs to.
    sample_metadata : dict
        Nested dictionary containing samples ID's and associated metadata.
    sources_envs : np.array
        Array of the sources in order of the columns in `sources_data`.
    output_dir : str
        Path to the output directory where the results will be saved.
    """
    sink_data = biom_table.data(sample, axis='sample', dense=True)
    _tmp = sample_metadata[sample][source_category]
    row = (sources_envs == _tmp).nonzero()[0]
    _sd = copy(sources_data)
    _sd[row] -= sink_data
    cp = ConditionalProbability(alpha1, alpha2, beta, _sd)
    results = gibbs_sampler(cp, sink_data, restarts, draws_per_restart,
                            burnin, delay)
    lines = _cli_single_sample_formatter(results[0])
    o = open(os.path.join(output_dir, sample + '.txt'), 'w')
    o.writelines(lines)
    o.close()


def _gibbs(source_df, sink_df, alpha1, alpha2, beta, restarts,
           draws_per_restart, burnin, delay, cluster=None):
    '''Private method for one-line calls of Gibb's sampling.

    Notes
    -----
    This function exists to allow one-line calls to source/sink prediction.
    This function currently does not support LOO classification. It is a
    private method meant to be used only by advanced users who know what they
    want to automate.

    Parameters that are not described in this function body are described
    elsewhere in this library (e.g. alpha1, alpha2, etc.)

    Warnings
    --------
    This function does _not_ perform rarefaction, the user should perform
    rarefaction prior to calling this function.

    Parameters
    ----------
    source_df : DataFrame
        A dataframe containing source data (rows are sources, columns are
        OTUs). The index must be the names of the sources.
    sink_df : DataFrame
        A dataframe containing sink data (rows are sinks, columns are OTUs).
        The index must be the names of the sinks.
    cluster : ipyparallel.client.client.Client or None
        An ipyparallel Client object, e.g. a started cluster.

    Returns
    -------
    mp : DataFrame
        A dataframe containing the mixing proportions (rows are sinks, columns
            are sources)
    mps : DataFrame
        A dataframe containing the mixing proportions standard deviations
        (rows are sinks, columns are sources)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ipyparallel import Client
    >>> import subprocess
    >>> import time
    >>> from sourcetracker.sourcetracker import \
        (_gibbs, _cli_sink_source_prediction_runner)

    Prepare some source data.
    >>> otus = np.array(['o%s' % i for i in range(50)])
    >>> source1 = np.random.randint(0, 1000, size=50)
    >>> source2 = np.random.randint(0, 1000, size=50)
    >>> source3 = np.random.randint(0, 1000, size=50)
    >>> source_df = pd.DataFrame([source1, source2, source3],
                                 index=['source1', 'source2', 'source3'],
                                 columns=otus, dtype=np.float64)

    Prepare some sink data.
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

    Set paramaters
    >>> alpha1 = .01
    >>> alpha2 = .001
    >>> beta = 10
    >>> restarts = 5
    >>> draws_per_restart = 1
    >>> burnin = 2
    >>> delay = 2

    Call without a cluster
    >>> mp, mps = _gibbs(source_df, sink_df, alpha1, alpha2, beta, restarts,
                         draws_per_restart, burnin, delay, cluster=None)

    Start a cluster and call the function.
    >>> jobs = 4
    >>> subprocess.Popen('ipcluster start -n %s --quiet' % jobs, shell=True)
    >>> time.sleep(25)
    >>> c = Client()
    >>> mp, mps = _gibbs(source_df, sink_df, alpha1, alpha2, beta, restarts,
                         draws_per_restart, burnin, delay, cluster=c)
    '''
    with TemporaryDirectory() as tmpdir:
        f = partial(_cli_sink_source_prediction_runner, alpha1=alpha1,
                    alpha2=alpha2, beta=beta, restarts=restarts,
                    draws_per_restart=draws_per_restart, burnin=burnin,
                    delay=delay, sources_data=source_df.values,
                    biom_table=sink_df, output_dir=tmpdir)
        if cluster is not None:
            cluster[:].map(f, sink_df.index, block=True)
        else:
            for sink in sink_df.index:
                f(sink)

        samples = []
        mp_means = []
        mp_stds = []
        for sample_fp in glob.glob(os.path.join(tmpdir, '*')):
            samples.append(sample_fp.strip().split('/')[-1].split('.txt')[0])
            tmp_arr = np.loadtxt(sample_fp, delimiter='\t')
            mp_means.append(tmp_arr.mean(0))
            mp_stds.append(tmp_arr.std(0))

    cols = list(source_df.index) + ['Unknown']
    mp_df = pd.DataFrame(mp_means, index=samples, columns=cols)
    mp_stds_df = pd.DataFrame(mp_stds, index=samples, columns=cols)
    return mp_df, mp_stds_df
