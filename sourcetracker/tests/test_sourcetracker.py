#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from __future__ import division

from unittest import TestCase, main

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sourcetracker._sourcetracker import (intersect_and_sort_samples,
                                          collapse_source_data,
                                          subsample_dataframe,
                                          validate_gibbs_input,
                                          validate_gibbs_parameters,
                                          collate_gibbs_results,
                                          get_samples,
                                          generate_environment_assignments,
                                          cumulative_proportions,
                                          single_sink_feature_table,
                                          ConditionalProbability,
                                          gibbs_sampler, gibbs,
                                          plot_heatmap)


class TestValidateGibbsInput(TestCase):

    def setUp(self):
        self.index = ['s%s' % i for i in range(5)]
        self.columns = ['f%s' % i for i in range(4)]

    def test_no_errors_(self):
        # A table where nothing is wrong, no changes expected.
        data = np.random.randint(0, 10, size=20).reshape(5, 4)
        sources = pd.DataFrame(data.astype(np.int32), index=self.index,
                               columns=self.columns)
        exp_sources = pd.DataFrame(data.astype(np.int32), index=self.index,
                                   columns=self.columns)
        obs = validate_gibbs_input(sources)
        pd.util.testing.assert_frame_equal(obs, sources)

        # Sources and sinks.
        sinks = pd.DataFrame(data, index=self.index, columns=self.columns)
        exp_sinks = pd.DataFrame(data.astype(np.int32), index=self.index,
                                 columns=self.columns)
        obs_sources, obs_sinks = validate_gibbs_input(sources, sinks)
        pd.util.testing.assert_frame_equal(obs_sources, exp_sources)
        pd.util.testing.assert_frame_equal(obs_sinks, exp_sinks)

    def test_float_data(self):
        # Data is float, expect rounding.
        data = np.random.uniform(0, 1, size=20).reshape(5, 4)
        sources = pd.DataFrame(data, index=self.index, columns=self.columns)
        exp_sources = pd.DataFrame(np.zeros(20).reshape(5, 4).astype(np.int32),
                                   index=self.index, columns=self.columns)
        obs_sources = validate_gibbs_input(sources)
        pd.util.testing.assert_frame_equal(obs_sources, exp_sources)

        data = np.random.uniform(0, 1, size=20).reshape(5, 4) + 1.
        sources = pd.DataFrame(data, index=self.index, columns=self.columns)
        exp_sources = pd.DataFrame(np.ones(20).reshape(5, 4).astype(np.int32),
                                   index=self.index, columns=self.columns)
        obs_sources = validate_gibbs_input(sources)
        pd.util.testing.assert_frame_equal(obs_sources, exp_sources)

        # Sources and sinks.
        data = np.random.uniform(0, 1, size=20).reshape(5, 4) + 5
        sinks = pd.DataFrame(data,
                             index=self.index,
                             columns=self.columns)
        exp_sinks = \
            pd.DataFrame(5 * np.ones(20).reshape(5, 4).astype(np.int32),
                         index=self.index,
                         columns=self.columns)
        obs_sources, obs_sinks = validate_gibbs_input(sources, sinks)
        pd.util.testing.assert_frame_equal(obs_sources, exp_sources)
        pd.util.testing.assert_frame_equal(obs_sinks, exp_sinks)

    def test_negative_data(self):
        # Values less than 0, expect errors.
        data = np.random.uniform(0, 1, size=20).reshape(5, 4) - 1.
        sources = pd.DataFrame(data,
                               index=self.index,
                               columns=self.columns)
        self.assertRaises(ValueError, validate_gibbs_input, sources)

        data = -1 * np.random.randint(0, 20, size=20).reshape(5, 4)
        sources = pd.DataFrame(data,
                               index=self.index,
                               columns=self.columns)
        self.assertRaises(ValueError, validate_gibbs_input, sources)

        # Sources and sinks.
        data = np.random.randint(0, 10, size=20).reshape(5, 4) + 1
        sources = pd.DataFrame(data.astype(np.int32),
                               index=self.index,
                               columns=self.columns)
        sinks = pd.DataFrame(-10 * data,
                             index=self.index,
                             columns=self.columns)
        self.assertRaises(ValueError, validate_gibbs_input, sources, sinks)

    def test_nan_data(self):
        # nans, expect errors.
        data = np.random.uniform(0, 1, size=20).reshape(5, 4)
        data[3, 2] = np.nan
        sources = pd.DataFrame(data,
                               index=self.index,
                               columns=self.columns)
        self.assertRaises(ValueError, validate_gibbs_input, sources)

        # Sources and sinks.
        data = np.random.randint(0, 10, size=20).reshape(5, 4) + 1.
        sources = pd.DataFrame(data,
                               index=self.index,
                               columns=self.columns)
        data[1, 3] = np.nan
        sinks = pd.DataFrame(data,
                             index=self.index,
                             columns=self.columns)
        self.assertRaises(ValueError, validate_gibbs_input, sources, sinks)

    def test_non_numeric_data(self):
        # data contains at least some non-numeric columns, expect errors.
        data = np.random.randint(0, 10, size=20).reshape(5, 4)
        sources = pd.DataFrame(data.astype(np.int32),
                               index=self.index,
                               columns=self.columns)
        sources.iloc[2, 2] = '3.a'
        self.assertRaises(ValueError, validate_gibbs_input, sources)

        # Sources and sinks.
        data = np.random.randint(0, 10, size=20).reshape(5, 4)
        sources = pd.DataFrame(data.astype(np.int32),
                               index=self.index,
                               columns=self.columns)
        sinks = pd.DataFrame(data.astype(np.int32),
                             index=self.index,
                             columns=self.columns)
        sinks.iloc[2, 2] = '3'
        self.assertRaises(ValueError, validate_gibbs_input, sources, sinks)

    def test_columns_identical(self):
        # Columns are identical, no error expected.
        data = np.random.randint(0, 10, size=20).reshape(5, 4)
        sources = pd.DataFrame(data.astype(np.int32),
                               index=self.index,
                               columns=self.columns)
        data = np.random.randint(0, 10, size=200).reshape(50, 4)
        sinks = pd.DataFrame(data.astype(np.int32),
                             index=['s%s' % i for i in range(50)],
                             columns=self.columns)
        obs_sources, obs_sinks = validate_gibbs_input(sources, sinks)
        pd.util.testing.assert_frame_equal(obs_sources, sources)
        pd.util.testing.assert_frame_equal(obs_sinks, sinks)

    def test_columns_non_identical(self):
        # Columns are not identical, error expected.
        data = np.random.randint(0, 10, size=20).reshape(5, 4)
        sources = pd.DataFrame(data.astype(np.int32),
                               index=self.index,
                               columns=self.columns)
        data = np.random.randint(0, 10, size=200).reshape(50, 4)
        sinks = pd.DataFrame(data.astype(np.int32),
                             index=['s%s' % i for i in range(50)],
                             columns=['feature%s' % i for i in range(4)])
        self.assertRaises(ValueError, validate_gibbs_input, sources, sinks)


class TestValidateGibbsParams(TestCase):

    def test_acceptable_inputs(self):
        # All values acceptable, expect no errors.
        alpha1 = .001
        alpha2 = .1
        beta = 10
        restarts = 10
        draws_per_restart = 1
        burnin = 100
        delay = 1
        self.assertTrue(validate_gibbs_parameters(alpha1, alpha2, beta,
                        restarts, draws_per_restart, burnin, delay))

        alpha1 = alpha2 = beta = 0
        self.assertTrue(validate_gibbs_parameters(alpha1, alpha2, beta,
                        restarts, draws_per_restart, burnin, delay))

    def test_not_acceptable_inputs(self):
        # One of the float params is negative.
        alpha1 = -.001
        alpha2 = .1
        beta = 10
        restarts = 10
        draws_per_restart = 1
        burnin = 100
        delay = 1
        self.assertFalse(validate_gibbs_parameters(alpha1, alpha2, beta,
                         restarts, draws_per_restart, burnin, delay))

        # One of the int params is 0.
        alpha1 = .001
        restarts = 0
        self.assertFalse(validate_gibbs_parameters(alpha1, alpha2, beta,
                         restarts, draws_per_restart, burnin, delay))

        # One of the int params is a float.
        restarts = 1.34
        self.assertFalse(validate_gibbs_parameters(alpha1, alpha2, beta,
                         restarts, draws_per_restart, burnin, delay))

        # A param is a string.
        restarts = '3.2232'
        self.assertFalse(validate_gibbs_parameters(alpha1, alpha2, beta,
                         restarts, draws_per_restart, burnin, delay))

        # A param is a nan.
        restarts = 3
        alpha1 = np.nan
        self.assertFalse(validate_gibbs_parameters(alpha1, alpha2, beta,
                         restarts, draws_per_restart, burnin, delay))


class TestIntersectAndSortSamples(TestCase):

    def test_partially_overlapping_tables(self):
        # Test an example where there are unshared samples present in both
        # feature and sample tables. Notice that order is different between
        # the samples that are shared between both tables. The order of samples
        # in the returned tables is set by the ordering done in np.intersect1d.
        sdata_c1 = [3.1, 'red', 5]
        sdata_c2 = [3.6, 'yellow', 7]
        sdata_c3 = [3.9, 'yellow', -2]
        sdata_c4 = [2.5, 'red', 5]
        sdata_c5 = [6.7, 'blue', 10]
        samples = ['s1', 's4', 's2', 's3', 'sX']
        headers = ['pH', 'color', 'day']
        stable = pd.DataFrame([sdata_c1, sdata_c4, sdata_c2, sdata_c3,
                               sdata_c5], index=samples, columns=headers)

        fdata = np.arange(90).reshape(9, 10)
        samples = ['s%i' % i for i in range(3, 12)]
        columns = ['o%i' % i for i in range(1, 11)]
        ftable = pd.DataFrame(fdata, index=samples, columns=columns)

        exp_ftable = pd.DataFrame(fdata[[1, 0], :], index=['s4', 's3'],
                                  columns=columns)
        exp_stable = pd.DataFrame([sdata_c4, sdata_c3], index=['s4', 's3'],
                                  columns=headers)

        obs_stable, obs_ftable = intersect_and_sort_samples(stable, ftable)

        pd.util.testing.assert_frame_equal(obs_stable, exp_stable)
        pd.util.testing.assert_frame_equal(obs_ftable, exp_ftable)

        # No shared samples, expect a ValueError.
        ftable.index = ['ss%i' % i for i in range(9)]
        self.assertRaises(ValueError, intersect_and_sort_samples, stable,
                          ftable)

        # All samples shared, expect no changes.
        fdata = np.arange(50).reshape(5, 10)
        samples = ['s1', 's4', 's2', 's3', 'sX']
        columns = ['o%i' % i for i in range(10)]
        ftable = pd.DataFrame(fdata, index=samples, columns=columns)

        exp_ftable = ftable.loc[stable.index, :]
        exp_stable = stable

        obs_stable, obs_ftable = intersect_and_sort_samples(stable, ftable)
        pd.util.testing.assert_frame_equal(obs_stable, exp_stable)
        pd.util.testing.assert_frame_equal(obs_ftable, exp_ftable)


class TestGetSamples(TestCase):

    def tests(self):
        # Make a dataframe which contains mixed data to test.
        col0 = ['a', 'a', 'a', 'a', 'b']
        col1 = [3, 2, 3, 1, 3]
        col2 = ['red', 'red', 'blue', 255, 255]
        headers = ['sample_location', 'num_reps', 'color']
        samples = ['s1', 's2', 's3', 's4', 's5']

        sample_metadata = \
            pd.DataFrame.from_dict({k: v for k, v in zip(headers,
                                                         [col0, col1, col2])})
        sample_metadata.index = samples

        obs = get_samples(sample_metadata, 'sample_location', 'b')
        exp = pd.Index(['s5'], dtype='object')
        pd.util.testing.assert_index_equal(obs, exp)

        obs = get_samples(sample_metadata, 'sample_location', 'a')
        exp = pd.Index(['s1', 's2', 's3', 's4'], dtype='object')
        pd.util.testing.assert_index_equal(obs, exp)

        obs = get_samples(sample_metadata, 'color', 255)
        exp = pd.Index(['s4', 's5'], dtype='object')
        pd.util.testing.assert_index_equal(obs, exp)

        obs = get_samples(sample_metadata, 'num_reps', 3)
        exp = pd.Index(['s1', 's3', 's5'], dtype='object')
        pd.util.testing.assert_index_equal(obs, exp)


class TestCollapseSourceData(TestCase):

    def test_example1(self):
        # Simple example with 'sum' as collapse mode.
        samples = ['sample1', 'sample2', 'sample3', 'sample4']
        category = 'pH'
        values = [3.0, 0.4, 3.0, 3.0]
        stable = pd.DataFrame(values, index=samples, columns=[category])
        fdata = np.array([[10,  50,  10,  70],
                          [0,  25,  10,   5],
                          [0,  25,  10,   5],
                          [100,   0,  10,   5]])
        ftable = pd.DataFrame(fdata, index=stable.index,
                              columns=map(str, np.arange(4)))
        source_samples = ['sample1', 'sample2', 'sample3']
        method = 'sum'
        obs = collapse_source_data(stable, ftable, source_samples, category,
                                   method)
        exp_data = np.vstack((fdata[1, :], fdata[0, :] + fdata[2, :]))
        exp_index = [0.4, 3.0]
        exp = pd.DataFrame(exp_data.astype(np.int32), index=exp_index,
                           columns=map(str, np.arange(4)))
        exp.index.name = 'collapse_col'
        pd.util.testing.assert_frame_equal(obs, exp)

        # Example with collapse mode 'mean'. This will cause non-integer values
        # to be present, which the validate_gibbs_input should catch.
        source_samples = ['sample1', 'sample2', 'sample3', 'sample4']
        method = 'mean'
        obs = collapse_source_data(stable, ftable, source_samples, category,
                                   method)
        exp_data = np.vstack((fdata[1, :],
                              fdata[[0, 2, 3], :].mean(0))).astype(np.int32)
        exp_index = [0.4, 3.0]
        exp = pd.DataFrame(exp_data.astype(np.int32), index=exp_index,
                           columns=map(str, np.arange(4)))
        exp.index.name = 'collapse_col'
        pd.util.testing.assert_frame_equal(obs, exp)

    def test_example2(self):
        # Test on another arbitrary example.
        data = np.arange(200).reshape(20, 10)
        oids = ['o%s' % i for i in range(20)]
        sids = ['s%s' % i for i in range(10)]
        ftable = pd.DataFrame(data.T, index=sids, columns=oids)
        _stable = \
            {'s4': {'cat1': '2', 'cat2': 'x', 'cat3': 'A', 'cat4': 'D'},
             's0': {'cat1': '1', 'cat2': 'y', 'cat3': 'z', 'cat4': 'D'},
             's1': {'cat1': '1', 'cat2': 'x', 'cat3': 'A', 'cat4': 'C'},
             's3': {'cat1': '2', 'cat2': 'y', 'cat3': 'z', 'cat4': 'A'},
             's2': {'cat1': '2', 'cat2': 'x', 'cat3': 'A', 'cat4': 'D'},
             's6': {'cat1': '1', 'cat2': 'y', 'cat3': 'z', 'cat4': 'R'},
             's5': {'cat1': '2', 'cat2': 'x', 'cat3': 'z', 'cat4': '0'},
             's7': {'cat1': '2', 'cat2': 'x', 'cat3': 'z', 'cat4': '0'},
             's9': {'cat1': '2', 'cat2': 'x', 'cat3': 'z', 'cat4': '0'},
             's8': {'cat1': '2', 'cat2': 'x', 'cat3': 'z', 'cat4': '0'}}
        stable = pd.DataFrame(_stable).T
        category = 'cat4'
        source_samples = ['s4', 's9', 's0', 's2']
        method = 'sum'
        obs = collapse_source_data(stable, ftable, source_samples, category,
                                   method)
        exp_index = np.array(['0', 'D'])
        exp_data = np.array([[9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119,
                              129, 139, 149, 159, 169, 179, 189, 199],
                             [6, 36, 66, 96, 126, 156, 186, 216, 246, 276, 306,
                              336,  366, 396, 426, 456, 486, 516, 546, 576]],
                            dtype=np.int32)

        exp = pd.DataFrame(exp_data, index=exp_index, columns=oids)
        exp.index.name = 'collapse_col'
        pd.util.testing.assert_frame_equal(obs, exp)


class TestSubsampleDataframe(TestCase):

    def test_no_errors_expected(self):
        # Testing this function deterministically is hard because cython is
        # generating the PRNG calls. We'll settle for ensuring that the sums
        # are correct.
        fdata = np.array([[10,  50,  10,  70],
                          [0,  25,  10,   5],
                          [0,  25,  10,   5],
                          [100,   0,  10,   5]])
        ftable = pd.DataFrame(fdata, index=['s1', 's2', 's3', 's4'],
                              columns=map(str, np.arange(4)))
        n = 30
        obs = subsample_dataframe(ftable, n)
        self.assertTrue((obs.sum(axis=1) == n).all())

    def test_shape_doesnt_change(self):
        # Test that when features are removed by subsampling, the shape of the
        # table does not change. Although rarifaction is stochastic, the
        # probability that the below table does not lose at least one feature
        # during rarefaction (and thus satisfy as the test of the condition we)
        # are interested in) is nearly 0.
        fdata = np.array([[0,   0,   0, 1e4],
                          [0,   0,   1, 1e4],
                          [0,   1,   0, 1e4],
                          [1,   0,   0, 1e4]]).astype(int)
        ftable = pd.DataFrame(fdata, index=['s1', 's2', 's3', 's4'],
                              columns=map(str, np.arange(4)))
        n = 10
        obs = subsample_dataframe(ftable, n)
        self.assertTrue((obs.sum(axis=1) == n).all())
        self.assertEqual(obs.shape, ftable.shape)


class TestDataAggregationFunctions(TestCase):
    '''Test that returned data is collated and written correctly.'''

    def test_cumulative_proportions(self):
        # 4 draws, 4 sources + unknown, 3 sinks
        sink1_envcounts = np.array([[10, 100, 15, 0, 25],
                                    [150, 0, 0, 0, 0],
                                    [30, 30, 30, 30, 30],
                                    [0, 11, 7, 35, 97]])
        sink2_envcounts = np.array([[100, 10, 15, 0, 25],
                                    [100, 0, 50, 0, 0],
                                    [0, 60, 30, 30, 30],
                                    [7, 11, 0, 35, 97]])
        sink3_envcounts = np.array([[100, 10, 10, 5, 25],
                                    [70, 20, 30, 30, 0],
                                    [10, 30, 50, 30, 30],
                                    [0, 27, 100, 20, 3]])
        all_envcounts = [sink1_envcounts, sink2_envcounts, sink3_envcounts]
        sink_ids = np.array(['sink1', 'sink2', 'sink3'])
        source_ids = np.array(['source1', 'source2', 'source3', 'source4'])
        cols = list(source_ids) + ['Unknown']

        prp_r1 = np.array([190, 141, 52, 65, 152]) / 600.
        prp_r2 = np.array([207, 81, 95, 65, 152]) / 600.
        prp_r3 = np.array([180, 87, 190, 85, 58]) / 600.
        prp_data = np.vstack([prp_r1, prp_r2, prp_r3])

        prp_std_data = np.zeros((3, 5), dtype=np.float64)
        prp_std_data[0, 0] = (np.array([10, 150, 30, 0]) / 600.).std()
        prp_std_data[0, 1] = (np.array([100, 0, 30, 11]) / 600.).std()
        prp_std_data[0, 2] = (np.array([15, 0, 30, 7]) / 600.).std()
        prp_std_data[0, 3] = (np.array([0, 0, 30, 35]) / 600.).std()
        prp_std_data[0, 4] = (np.array([25, 0, 30, 97]) / 600.).std()

        prp_std_data[1, 0] = (np.array([100, 100, 0, 7]) / 600.).std()
        prp_std_data[1, 1] = (np.array([10, 0, 60, 11]) / 600.).std()
        prp_std_data[1, 2] = (np.array([15, 50, 30, 0]) / 600.).std()
        prp_std_data[1, 3] = (np.array([0, 0, 30, 35]) / 600.).std()
        prp_std_data[1, 4] = (np.array([25, 0, 30, 97]) / 600.).std()

        prp_std_data[2, 0] = (np.array([100, 70, 10, 0]) / 600.).std()
        prp_std_data[2, 1] = (np.array([10, 20, 30, 27]) / 600.).std()
        prp_std_data[2, 2] = (np.array([10, 30, 50, 100]) / 600.).std()
        prp_std_data[2, 3] = (np.array([5, 30, 30, 20]) / 600.).std()
        prp_std_data[2, 4] = (np.array([25, 0, 30, 3]) / 600.).std()

        exp_prp = pd.DataFrame(prp_data, index=sink_ids, columns=cols)
        exp_prp_std = pd.DataFrame(prp_std_data, index=sink_ids, columns=cols)
        obs_prp, obs_prp_std = cumulative_proportions(all_envcounts, sink_ids,
                                                      source_ids)
        pd.util.testing.assert_frame_equal(obs_prp, exp_prp)
        pd.util.testing.assert_frame_equal(obs_prp_std, exp_prp_std)

    def test_single_sink_feature_table(self):
        # 4 draws, depth of sink = 10, 5 sources + Unknown.
        final_env_assignments = np.array([[5, 0, 0, 0, 2, 0, 1, 0, 3, 1],
                                          [1, 1, 3, 3, 2, 2, 1, 1, 1, 1],
                                          [4, 1, 4, 4, 4, 4, 1, 1, 3, 2],
                                          [2, 1, 0, 5, 5, 5, 5, 1, 0, 2]])
        # notice that each row is the same - they are determined by
        # `generate_taxon_sequence` before the `gibbs_sampler` runs.
        final_taxon_assignments = \
            np.array([[0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100]])
        # we are allowing more taxa than we have found in this sample, i.e. the
        # largest value in `final_taxon_assignments` will be smaller than the
        # largest index in the columns of the final table.
        nfeatures = 1250
        nsources = 5
        data = np.zeros((nsources + 1, nfeatures), dtype=np.int32)

        # for the purpose of this test code, I'll increment data taxa by taxa.
        data[np.array([5, 1, 4, 2]), 0] += 1
        data[0, 3] += 3
        data[1, 3] += 3
        data[3, 3] += 1
        data[4, 3] += 1
        data[np.array([0, 3, 4, 5]), 227] += 1
        data[0, 550] += 1
        data[1, 550] += 3
        data[2, 550] += 3
        data[4, 550] += 2
        data[5, 550] += 3
        data[0, 999] += 2
        data[1, 999] += 4
        data[3, 999] += 2
        data[1, 1100] += 2
        data[2, 1100] += 2

        exp_sources = ['source%s' % i for i in range(nsources)] + ['Unknown']
        feature_ids = ['f%s' % i for i in range(1250)]
        exp = pd.DataFrame(data, index=exp_sources, columns=feature_ids)

        source_ids = np.array(['source%s' % i for i in range(nsources)])
        obs = single_sink_feature_table(final_env_assignments,
                                        final_taxon_assignments, source_ids,
                                        feature_ids)

        pd.util.testing.assert_frame_equal(obs, exp)

    def test_collate_gibbs_results(self):
        # We'll vary the depth of the sinks - simulating a situation where the
        # user has not rarefied.
        # We'll set:
        # draws = 4
        # sink_depths = [10, 15, 7]
        # sources = 5 (+1 unknown)
        final_env_counts_sink1 = np.array([[5, 2, 1, 1, 0, 1],
                                           [0, 6, 2, 2, 0, 0],
                                           [0, 3, 1, 1, 5, 0],
                                           [2, 2, 2, 0, 0, 4]])
        final_env_assignments_sink1 = \
            np.array([[5, 0, 0, 0, 2, 0, 1, 0, 3, 1],
                      [1, 1, 3, 3, 2, 2, 1, 1, 1, 1],
                      [4, 1, 4, 4, 4, 4, 1, 1, 3, 2],
                      [2, 1, 0, 5, 5, 5, 5, 1, 0, 2]])
        final_taxon_assignments_sink1 = \
            np.array([[0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100],
                      [0, 3, 3, 227, 550, 550, 550, 999, 999, 1100]])

        final_env_counts_sink2 = np.array([[5, 1, 3, 2, 0, 4],
                                           [1, 1, 4, 5, 1, 3],
                                           [4, 1, 3, 2, 3, 2],
                                           [2, 3, 3, 2, 1, 4]])
        final_env_assignments_sink2 = \
            np.array([[2, 5, 0, 5, 1, 5, 0, 0, 3, 0, 3, 5, 2, 2, 0],
                      [3, 2, 2, 3, 2, 3, 3, 5, 5, 1, 3, 4, 2, 0, 5],
                      [0, 2, 3, 2, 0, 0, 2, 4, 5, 4, 0, 5, 3, 1, 4],
                      [4, 3, 2, 1, 2, 5, 3, 5, 2, 0, 1, 0, 5, 1, 5]])
        final_taxon_assignments_sink2 = \
            np.array([[7, 7, 7, 7, 8, 8, 8, 8, 250, 250, 250, 250, 1249, 1249],
                      [7, 7, 7, 7, 8, 8, 8, 8, 250, 250, 250, 250, 1249, 1249],
                      [7, 7, 7, 7, 8, 8, 8, 8, 250, 250, 250, 250, 1249, 1249],
                     [7, 7, 7, 7, 8, 8, 8, 8, 250, 250, 250, 250, 1249, 1249]])

        final_env_counts_sink3 = np.array([[4, 2, 0, 0, 1, 0],
                                           [0, 3, 1, 0, 2, 1],
                                           [0, 0, 1, 1, 3, 2],
                                           [2, 1, 0, 3, 0, 1]])
        final_env_assignments_sink3 = \
            np.array([[4, 0, 0, 0, 1, 0, 1],
                      [1, 2, 1, 4, 5, 4, 1],
                      [4, 3, 5, 4, 4, 5, 2],
                      [3, 0, 1, 3, 3, 0, 5]])
        final_taxon_assignments_sink3 = \
            np.array([[3, 865, 865, 1100, 1100, 1100, 1249],
                      [3, 865, 865, 1100, 1100, 1100, 1249],
                      [3, 865, 865, 1100, 1100, 1100, 1249],
                      [3, 865, 865, 1100, 1100, 1100, 1249]])

        # Create expected proportion data.
        prp_data = np.zeros((3, 6), dtype=np.float64)
        prp_std_data = np.zeros((3, 6), dtype=np.float64)

        prp_data[0] = (final_env_counts_sink1.sum(0) /
                       final_env_counts_sink1.sum())
        prp_data[1] = (final_env_counts_sink2.sum(0) /
                       final_env_counts_sink2.sum())
        prp_data[2] = (final_env_counts_sink3.sum(0) /
                       final_env_counts_sink3.sum())

        prp_std_data[0] = \
            (final_env_counts_sink1 / final_env_counts_sink1.sum()).std(0)
        prp_std_data[1] = \
            (final_env_counts_sink2 / final_env_counts_sink2.sum()).std(0)
        prp_std_data[2] = \
            (final_env_counts_sink3 / final_env_counts_sink3.sum()).std(0)

        sink_ids = ['sink1', 'sink2', 'sink3']
        exp_sources = ['source%s' % i for i in range(5)] + ['Unknown']
        feature_ids = ['f%s' % i for i in range(1250)]

        exp_prp = pd.DataFrame(prp_data, index=sink_ids, columns=exp_sources)
        exp_prp_std = pd.DataFrame(prp_std_data, index=sink_ids,
                                   columns=exp_sources)

        # Create expected feature table data.
        ft1 = np.zeros((6, 1250), dtype=np.int32)
        for r, c in zip(final_env_assignments_sink1.ravel(),
                        final_taxon_assignments_sink1.ravel()):
            ft1[r, c] += 1
        exp_ft1 = pd.DataFrame(ft1, index=exp_sources, columns=feature_ids)
        ft2 = np.zeros((6, 1250), dtype=np.int32)
        for r, c in zip(final_env_assignments_sink2.ravel(),
                        final_taxon_assignments_sink2.ravel()):
            ft2[r, c] += 1
        exp_ft2 = pd.DataFrame(ft2, index=exp_sources, columns=feature_ids)
        ft3 = np.zeros((6, 1250), dtype=np.int32)
        for r, c in zip(final_env_assignments_sink3.ravel(),
                        final_taxon_assignments_sink3.ravel()):
            ft3[r, c] += 1
        exp_ft3 = pd.DataFrame(ft3, index=exp_sources, columns=feature_ids)
        exp_fts = [exp_ft1, exp_ft2, exp_ft3]

        # Prepare the inputs for passing to collate_gibbs_results
        all_envcounts = [final_env_counts_sink1, final_env_counts_sink2,
                         final_env_counts_sink3]
        all_env_assignments = [final_env_assignments_sink1,
                               final_env_assignments_sink2,
                               final_env_assignments_sink3]
        all_taxon_assignments = [final_taxon_assignments_sink1,
                                 final_taxon_assignments_sink2,
                                 final_taxon_assignments_sink3]

        # Test when create_feature_tables=True
        obs_prp, obs_prp_std, obs_fts = \
            collate_gibbs_results(all_envcounts, all_env_assignments,
                                  all_taxon_assignments, np.array(sink_ids),
                                  np.array(exp_sources[:-1]),
                                  np.array(feature_ids),
                                  create_feature_tables=True, loo=False)
        pd.util.testing.assert_frame_equal(obs_prp, exp_prp)
        pd.util.testing.assert_frame_equal(obs_prp_std, exp_prp_std)
        for i in range(3):
            pd.util.testing.assert_frame_equal(obs_fts[i], exp_fts[i])

        # Test when create_feature_tables=False
        obs_prp, obs_prp_std, obs_fts = \
            collate_gibbs_results(all_envcounts, all_env_assignments,
                                  all_taxon_assignments, np.array(sink_ids),
                                  np.array(exp_sources[:-1]),
                                  np.array(feature_ids),
                                  create_feature_tables=False, loo=False)
        self.assertTrue(obs_fts is None)

    def test_collate_gibbs_results_loo(self):
        # We'll vary the depth of the sources - simulating a situation where
        # the user has not rarefied.
        # We'll set:
        # draws = 2
        # source_depths = [7, 4, 5]
        # sources = 3 (+1 Unknown)
        ec1 = np.array([[6, 0, 1],
                        [2, 2, 3]])
        ea1 = np.array([[0, 2, 0, 0, 0, 0, 0],
                        [0, 1, 0, 2, 1, 2, 2]])
        ta1 = np.array([[2, 2, 2, 4, 4, 4, 6],
                        [2, 2, 2, 4, 4, 4, 6]])

        ec2 = np.array([[1, 2, 1],
                        [2, 2, 0]])
        ea2 = np.array([[0, 1, 2, 1],
                        [0, 1, 1, 0]])
        ta2 = np.array([[3, 3, 3, 3],
                        [3, 3, 3, 3]])

        ec3 = np.array([[1, 2, 2],
                        [4, 0, 1]])
        ea3 = np.array([[1, 1, 0, 2, 2],
                        [0, 0, 0, 0, 2]])
        ta3 = np.array([[3, 3, 4, 5, 5],
                        [3, 3, 4, 5, 5]])

        # Create expected proportion data.
        prp_data = np.array([[0, 8/14., 2/14., 4/14.],
                             [3/8., 0, 4/8., 1/8.],
                             [5/10., 2/10., 0, 3/10.]], dtype=np.float64)
        prp_std_data = np.zeros((3, 4), dtype=np.float64)

        prp_std_data[0, 1:] = (ec1 / ec1.sum()).std(0)
        prp_std_data[1, np.array([0, 2, 3])] = (ec2 / ec2.sum()).std(0)
        prp_std_data[2, np.array([0, 1, 3])] = (ec3 / ec3.sum()).std(0)

        exp_sources = ['source%s' % i for i in range(3)] + ['Unknown']
        feature_ids = ['f%s' % i for i in range(7)]

        exp_prp = pd.DataFrame(prp_data, index=exp_sources[:-1],
                               columns=exp_sources)
        exp_prp_std = pd.DataFrame(prp_std_data, index=exp_sources[:-1],
                                   columns=exp_sources)

        # Create expected feature table data.
        ft1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 4, 0, 3, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 2, 0, 1]], dtype=np.int64)
        ft2 = np.array([[0, 0, 0, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0]], dtype=np.int64)
        ft3 = np.array([[0, 0, 0, 2, 2, 1, 0],
                        [0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 3, 0]], dtype=np.int64)
        exp_fts = [pd.DataFrame(ft1, index=exp_sources, columns=feature_ids),
                   pd.DataFrame(ft2, index=exp_sources, columns=feature_ids),
                   pd.DataFrame(ft3, index=exp_sources, columns=feature_ids)]

        # Prepare the inputs for passing to collate_gibbs_results
        all_envcounts = [ec1, ec2, ec3]
        all_env_assignments = [ea1, ea2, ea3]
        all_taxon_assignments = [ta1, ta2, ta3]

        # Test when create_feature_tables=True
        obs_prp, obs_prp_std, obs_fts = \
            collate_gibbs_results(all_envcounts, all_env_assignments,
                                  all_taxon_assignments,
                                  np.array(exp_sources[:-1]),
                                  np.array(exp_sources[:-1]),
                                  np.array(feature_ids),
                                  create_feature_tables=True, loo=True)

        pd.util.testing.assert_frame_equal(obs_prp, exp_prp)
        pd.util.testing.assert_frame_equal(obs_prp_std, exp_prp_std)
        for i in range(3):
            pd.util.testing.assert_frame_equal(obs_fts[i], exp_fts[i])

        # Test when create_feature_tables=False
        obs_prp, obs_prp_std, obs_fts = \
            collate_gibbs_results(all_envcounts, all_env_assignments,
                                  all_taxon_assignments,
                                  np.array(exp_sources[:-1]),
                                  np.array(exp_sources[:-1]),
                                  np.array(feature_ids),
                                  create_feature_tables=False, loo=True)
        self.assertTrue(obs_fts is None)


class TestBookkeeping(TestCase):
    '''Tests for fnxs which generate bookkeeping data for `gibbs_sampler`.'''

    def test_generate_environment_assignment(self):
        np.random.seed(235234234)
        obs_sea, obs_ecs = generate_environment_assignments(100, 10)
        exp_sea = \
            np.array([7, 3, 4, 1, 5, 2, 6, 3, 6, 4, 4, 7, 8, 2, 7, 7, 9, 9, 4,
                      7, 0, 3, 6, 5, 7, 2, 7, 1, 2, 4, 1, 7, 0, 7, 5, 2, 8, 5,
                      3, 3, 1, 4, 3, 3, 8, 7, 7, 5, 2, 6, 0, 2, 4, 0, 0, 5, 9,
                      8, 2, 8, 9, 9, 8, 7, 5, 8, 0, 9, 8, 6, 3, 2, 3, 7, 3, 8,
                      4, 4, 9, 1, 6, 6, 0, 9, 2, 9, 9, 4, 2, 9, 0, 4, 1, 3, 4,
                      0, 0, 9, 8, 3])
        exp_ecs = np.array([10,  6, 11, 12, 12,  7,  7, 13, 10, 12])
        np.testing.assert_array_equal(obs_sea, exp_sea)
        np.testing.assert_array_equal(obs_ecs, exp_ecs)


class ConditionalProbabilityTests(TestCase):
    '''Unit test for the ConditionalProbability class.'''

    def setUp(self):
        # create an object we can reuse for several of the tests
        self.alpha1 = .5
        self.alpha2 = .001
        self.beta = 10
        self.source_data = np.array([[0, 0, 0, 100, 100, 100],
                                     [100, 100, 100, 0, 0, 0]])
        self.cp = ConditionalProbability(self.alpha1, self.alpha2, self.beta,
                                         self.source_data)

    def test_init(self):
        exp_alpha1 = self.alpha1
        exp_alpha2 = self.alpha2
        exp_beta = self.beta
        exp_m_xivs = self.source_data
        exp_m_vs = np.array([[300], [300]])
        exp_V = 3
        exp_tau = 6
        exp_joint_probability = np.array([0, 0, 0])

        self.assertEqual(self.cp.alpha1, exp_alpha1)
        self.assertEqual(self.cp.alpha2, exp_alpha2)
        self.assertEqual(self.cp.beta, exp_beta)
        np.testing.assert_array_equal(self.cp.m_xivs, exp_m_xivs)
        np.testing.assert_array_equal(self.cp.m_vs, exp_m_vs)
        self.assertEqual(self.cp.V, exp_V)
        self.assertEqual(self.cp.tau, exp_tau)
        np.testing.assert_array_equal(self.cp.joint_probability,
                                      exp_joint_probability)

    def test_set_n(self):
        self.cp.set_n(500)
        self.assertEqual(self.cp.n, 500)

    def test_precalculate(self):
        alpha1 = .01
        alpha2 = .3
        beta = 35
        source_data = np.array([[10, 5,  2,  100],
                                [0,  76, 7,  3],
                                [9,  5,  0,  0],
                                [0,  38, 11, 401]])
        cp = ConditionalProbability(alpha1, alpha2, beta, source_data)
        n = 1300
        cp.set_n(n)
        cp.precalculate()

        # Calculated by hand.
        exp_known_p_tv = np.array(
            [[.085526316, .042805878, .017173636, .85449419],
             [.000116225, .883426313, .081473733, .034983728],
             [.641737892, .356837607, .000712251, .000712251],
             [.00002222, .084459159, .024464492, .891054129]])
        exp_denominator_p_v = 1299 + 35 * 5
        exp_known_source_cp = exp_known_p_tv / exp_denominator_p_v
        exp_alpha2_n = 390
        exp_alpha2_n_tau = 1560

        self.assertEqual(cp.denominator_p_v, exp_denominator_p_v)
        self.assertEqual(cp.alpha2_n, exp_alpha2_n)
        self.assertEqual(cp.alpha2_n_tau, exp_alpha2_n_tau)
        np.testing.assert_array_almost_equal(cp.known_p_tv, exp_known_p_tv)
        np.testing.assert_array_almost_equal(cp.known_source_cp,
                                             exp_known_source_cp)

    def test_calculate_cp_slice(self):
        # test with non overlapping two component mixture.
        n = 500
        self.cp.set_n(n)
        self.cp.precalculate()

        n_vnoti = np.array([305, 1, 193])
        m_xiVs = np.array([25, 30, 29, 10, 60, 39])
        m_V = 193  # == m_xiVs.sum() == n_vnoti[2]

        # Calculated by hand.
        exp_jp_array = np.array(
            [[9.82612e-4, 9.82612e-4, 9.82612e-4, .1975051, .1975051,
              .1975051],
             [6.897003e-3, 6.897003e-3, 6.897003e-3, 3.4313e-5, 3.4313e-5,
              3.4313e-5],
             [.049925736, .059715096, .057757224, .020557656, .118451256,
              .077335944]])

        obs_jp_array = np.zeros((3, 6))
        for i in range(6):
            obs_jp_array[:, i] = self.cp.calculate_cp_slice(i, m_xiVs[i], m_V,
                                                            n_vnoti)

        np.testing.assert_array_almost_equal(obs_jp_array, exp_jp_array)

        # Test using Dan's R code and some print statements. Using the same
        # data as specified in setup.
        # Print statesments are added starting at line 347 of SourceTracker.r.
        # The output is being used to compare the p_tv * p_v calculation that
        # we are making. Used the following print statements:
        # print(sink)
        # print(taxon)
        # print(sources)
        # print(rowSums(sources))
        # print(envcounts)
        # print(p_v_denominator)
        # print('')
        # print(p_tv)
        # print(p_v)
        # print(p_tv * p_v)

        # Results of print statements
        # [1] 6
        # [1] 100 100 100 100 100 100
        #          otu_1 otu_2 otu_3 otu_4 otu_5 otu_6
        # Source_1   0.5   0.5   0.5 100.5 100.5 100.5
        # Source_2 100.5 100.5 100.5   0.5   0.5   0.5
        # Unknown   36.6  29.6  29.6  37.6  26.6  31.6
        # Source_1 Source_2  Unknown
        #    303.0    303.0    191.6
        # [1] 213 218 198
        # [1] 629
        # [1] ""
        #    Source_1    Source_2     Unknown
        # 0.331683168 0.001650165 0.164926931
        # [1] 0.3386328 0.3465819 0.3147854
        #     Source_1     Source_2      Unknown
        # 0.1123187835 0.0005719173 0.0519165856

        # The sink is the sum of the source data, self.source_data.sum(1).
        cp = ConditionalProbability(self.alpha1, self.alpha2, self.beta,
                                    self.source_data)
        cp.set_n(600)
        cp.precalculate()

        # Taxon selected by R was 6, but R is 1-indexed and python is
        # 0-indexed.
        taxon_index = 5
        # Must subtract alpha2 * tau * n from the Unknown sum since the R
        # script adds these values to the 'Sources' matrix.
        unknown_sum = 188
        unknown_at_t5 = 31
        # Must subtract beta from each envcount because the R script adds this
        # value to the 'envcounts' matrix.
        envcounts = np.array([203, 208, 188])
        obs_jp = cp.calculate_cp_slice(taxon_index, unknown_at_t5, unknown_sum,
                                       envcounts)
        # From the final line of R results above.
        exp_jp = np.array([0.1123187835, 0.0005719173, 0.0519165856])

        np.testing.assert_array_almost_equal(obs_jp, exp_jp)


class TestGibbs(TestCase):
    '''Unit tests for Gibbs based on seeding the PRNG and hand calculations.'''

    def test_single_pass_gibbs_sampler(self):
        # The data for this test was built by seeding the PRNG, and making the
        # calculations that Gibb's would make, and then comparing the results.
        restarts = 1
        draws_per_restart = 1
        burnin = 0
        # Setting delay to 2 is the only way to stop the Sampler after a single
        # pass.
        delay = 2
        alpha1 = .2
        alpha2 = .1
        beta = 3
        source_data = np.array([[0, 1, 4, 10],
                                [3, 2, 1, 1]])
        sink = np.array([2, 1, 4, 2])

        # Make calculations using gibbs function.
        np.random.seed(0)
        cp = ConditionalProbability(alpha1, alpha2, beta, source_data)
        obs_ec, obs_ea, obs_ta = gibbs_sampler(sink, cp, restarts,
                                               draws_per_restart, burnin,
                                               delay)

        # Make calculation using handrolled.
        np.random.seed(0)
        choices = np.arange(3)
        np.random.choice(choices, size=9, replace=True)
        order = np.arange(9)
        np.random.shuffle(order)
        expected_et_pairs = np.array([[2, 0, 1, 2, 0, 1, 0, 1, 0],
                                      [3, 2, 2, 2, 0, 0, 1, 2, 3]])
        envcounts = np.array([4., 3., 2.])
        unknown_vector = np.array([0, 0, 1, 1])
        # Calculate known probabilty base as ConditionalProbability would.
        denominator = np.array([[(15 + (4*.2)) * (8 + 3*3)],
                                [(7 + (4*.2)) * (8 + 3*3)]])
        numerator = np.array([[0, 1, 4, 10],
                              [3, 2, 1, 1]]) + .2
        known_env_prob_base = numerator / denominator

        # Set up a sequence environment assignments vector. This would normally
        # be handled by the Sampler class.
        seq_env_assignments = np.zeros(9)

        # Set up joint probability holder, normally handeled by
        # ConditionalProbability class.
        joint_prob = np.zeros(3)

        for i, (e, t) in enumerate(expected_et_pairs.T):
            envcounts[e] -= 1
            if e == 2:
                unknown_vector[t] -= 1
            # Calculate the new probabilty as ConditionalProbability would.
            joint_prob = np.zeros(3)
            joint_prob[:-1] += envcounts[:-1] + beta
            joint_prob[:-1] = joint_prob[:-1] * known_env_prob_base[:2, t]
            joint_prob[-1] = (unknown_vector[t] + (9 * .1)) / \
                             (unknown_vector.sum() + (9 * .1 * 4))
            joint_prob[-1] *= ((envcounts[2] + beta) / (8 + 3*3))

            # Another call to the PRNG
            new_e = np.random.choice(np.array([0, 1, 2]),
                                     p=joint_prob/joint_prob.sum())
            seq_env_assignments[i] = new_e
            envcounts[new_e] += 1
            if new_e == 2:
                unknown_vector[t] += 1

        # prps = envcounts / float(envcounts.sum())
        # exp_mps = prps/prps.sum()
        # Create taxon table like Sampler class would.
        exp_ct = np.zeros((4, 3))
        for i in range(9):
            exp_ct[expected_et_pairs[1, i],
                   np.int(seq_env_assignments[i])] += 1

        # np.testing.assert_array_almost_equal(obs_mps.squeeze(), exp_mps)
        # np.testing.assert_array_equal(obs_ct.squeeze().T, exp_ct)

        np.testing.assert_array_equal(obs_ec.squeeze(), envcounts)
        np.testing.assert_array_equal(obs_ea.squeeze()[order],
                                      seq_env_assignments)
        np.testing.assert_array_equal(obs_ta.squeeze()[order],
                                      expected_et_pairs[1, :])

    def test_gibbs_params_bad(self):
        # test gibbs when the parameters passed are bad
        features = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
        source1 = np.array([10, 10, 10, 0, 0, 0])
        source2 = np.array([0, 0, 0, 10, 10, 10])
        sources = pd.DataFrame(np.vstack((source1, source2)).astype(np.int32),
                               index=['source1', 'source2'], columns=features)
        self.assertRaises(ValueError, gibbs, sources, alpha1=-.3)

    def test_consistency_when_gibbs_seeded(self):
        '''Test consistency of `gibbs` (without LOO) from run to run.

        Notes
        -----
        The number of calls to the PRNG should be stable (and thus this test,
        which is seeded, should not fail). Any changes made to the code which
        cause this test to fail should be scrutinized very carefully.

        If the number of calls to the PRNG has not been changed, then an error
        has been introduced somewhere else in the code. If the number of calls
        has been changed, the deterministic tests should fail as well, but
        since they are a small example they might not fail (false negative).
        This test is extensive (it does 201 loops through the entire
        `gibbs_sampler` block).
        '''
        features = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
        source1 = np.array([10, 10, 10, 0, 0, 0])
        source2 = np.array([0, 0, 0, 10, 10, 10])
        sink1 = .5*source1 + .5*source2
        sinks = pd.DataFrame(sink1.reshape(1, 6).astype(np.int32),
                             index=['sink1'], columns=features)
        sources = pd.DataFrame(np.vstack((source1, source2)).astype(np.int32),
                               index=['source1', 'source2'], columns=features)

        np.random.seed(1042)
        mpm, mps, fts = gibbs(sources, sinks, alpha1=.001, alpha2=.01, beta=1,
                              restarts=3, draws_per_restart=5, burnin=50,
                              delay=4, cluster=None,
                              create_feature_tables=True)

        possible_sources = ['source1', 'source2', 'Unknown']
        vals = np.array([[0.44, 0.44666667, 0.11333333]])
        exp_mpm = pd.DataFrame(vals, index=['sink1'], columns=possible_sources)

        vals = np.array([[0.00824322, 0.00435465, 0.01047985]])
        exp_mps = pd.DataFrame(vals, index=['sink1'], columns=possible_sources)

        vals = np.array([[69, 64, 65,  0,  0,  0],
                         [0, 0,  0, 67, 70, 64],
                         [6, 11, 10,  8, 5, 11]], dtype=np.int32)
        exp_fts = pd.DataFrame(vals, index=possible_sources, columns=features)

        pd.util.testing.assert_frame_equal(mpm, exp_mpm)
        pd.util.testing.assert_frame_equal(mps, exp_mps)
        pd.util.testing.assert_frame_equal(fts[0], exp_fts)

    def test_consistency_when_gibbs_loo_seeded(self):
        '''Test consistency of `gibbs` (loo) from run to run.

        Notes
        -----
        The number of calls to the PRNG should be stable (and thus this test,
        which is seeded, should not fail). Any changes made to the code which
        cause this test to fail should be scrutinized very carefully.

        If the number of calls to the PRNG has not been changed, then an error
        has been introduced somewhere else in the code. If the number of calls
        has been changed, the deterministic tests should fail as well, but
        since they are a small example they might not fail (false negative).
        This test is extensive (it does 201 loops through the entire
        `gibbs_sampler` block for each source).
        '''
        source1a = np.array([10, 10, 10, 0, 0, 0])
        source1b = np.array([8, 8, 8, 2, 2, 2])
        source2a = np.array([0, 0, 0, 10, 10, 10])
        source2b = np.array([4, 4, 4, 6, 6, 6])

        vals = np.vstack((source1a, source1b, source2a,
                          source2b)).astype(np.int32)
        source_names = ['source1a', 'source1b', 'source2a', 'source2b']
        feature_names = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6']
        sources = pd.DataFrame(vals, index=source_names, columns=feature_names)

        np.random.seed(1042)
        obs_mpm, obs_mps, obs_fts = gibbs(sources, sinks=None, alpha1=.001,
                                          alpha2=.01, beta=1, restarts=3,
                                          draws_per_restart=5, burnin=50,
                                          delay=4, cluster=None,
                                          create_feature_tables=True)

        vals = np.array([[0., 0.62444444, 0., 0.01555556, 0.36],
                         [0.68444444, 0., 0.09333333, 0.12666667, 0.09555556],
                         [0., 0.00888889, 0., 0.08222222, 0.90888889],
                         [0.19111111, 0.2, 0.5, 0., 0.10888889]])
        exp_mpm = pd.DataFrame(vals, index=source_names,
                               columns=source_names + ['Unknown'])

        vals = np.array([[0., 0.02406393, 0., 0.0015956, 0.02445387],
                         [0.0076923, 0., 0.00399176, 0.00824322, 0.00648476],
                         [0., 0.00127442, 0., 0.00622575, 0.00609752],
                         [0.00636175, 0.00786721, 0.00525874, 0., 0.00609752]])
        exp_mps = pd.DataFrame(vals, index=source_names,
                               columns=source_names + ['Unknown'])

        fts0_vals = np.array([[0, 0, 0, 0, 0, 0],
                              [93, 87, 101, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [3, 4, 0, 0, 0, 0],
                              [54, 59, 49, 0, 0, 0]])
        fts1_vals = np.array([[113, 98, 97, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 15, 13, 14],
                              [5, 7, 11, 11, 12, 11],
                              [2, 15, 12, 4, 5, 5]])
        fts2_vals = np.array([[0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 2, 1, 1],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 12, 12, 13],
                              [0, 0, 0, 136, 137, 136]])
        fts3_vals = np.array([[28, 27, 31, 0, 0, 0],
                              [27, 24, 25, 3, 4, 7],
                              [0, 0, 0, 80, 71, 74],
                              [0, 0, 0, 0, 0, 0],
                              [5, 9, 4, 7, 15, 9]])
        fts_vals = [fts0_vals, fts1_vals, fts2_vals, fts3_vals]
        exp_fts = [pd.DataFrame(vals, index=source_names + ['Unknown'],
                   columns=feature_names) for vals in fts_vals]

        pd.util.testing.assert_frame_equal(obs_mpm, exp_mpm)
        pd.util.testing.assert_frame_equal(obs_mps, exp_mps)
        for obs_fts, exp_fts in zip(obs_fts, exp_fts):
            pd.util.testing.assert_frame_equal(obs_fts, exp_fts)

    def test_gibbs_close_to_sourcetracker_1(self):
        '''This test is stochastic; occasional errors might occur.

        Notes
        -----
        This tests against the R-code SourceTracker version 1.0, using
        R version 2.15.3.
        '''

        sources_data = \
            np.array([[0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  4],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0,  0]],
                     dtype=np.int32)
        sources_names = ['source1', 'source2', 'source3']
        feature_names = ['f%i' % i for i in range(32)]
        sources = pd.DataFrame(sources_data, index=sources_names,
                               columns=feature_names)

        sinks_data = np.array([[0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 385, 0, 0, 0, 0, 0, 0, 0, 350, 0, 0,
                                0, 0, 95],
                               [0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 380, 0, 0, 0, 0, 0, 0, 0, 350, 0, 0,
                                0, 0, 100],
                               [0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 378, 0, 0, 0, 0, 0, 0, 0, 350, 0, 0,
                                0, 0, 102],
                               [0, 0, 0, 0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 386, 0, 0, 0, 0, 0, 0, 0, 350, 0, 0,
                                0, 0, 94]], dtype=np.int32)
        sinks_names = ['sink1', 'sink2', 'sink3', 'sink4']
        sinks = pd.DataFrame(sinks_data, index=sinks_names,
                             columns=feature_names)

        obs_mpm, obs_mps, _ = gibbs(sources, sinks, alpha1=.001, alpha2=.1,
                                    beta=10, restarts=2, draws_per_restart=2,
                                    burnin=5, delay=2, cluster=None,
                                    create_feature_tables=False)

        exp_vals = np.array([[0.1695, 0.4781, 0.3497, 0.0027],
                             [0.1695, 0.4794, 0.3497, 0.0014],
                             [0.1693, 0.4784, 0.3499, 0.0024],
                             [0.1696, 0.4788, 0.3494, 0.0022]])
        exp_mpm = pd.DataFrame(exp_vals, index=sinks_names,
                               columns=sources_names + ['Unknown'])

        pd.util.testing.assert_index_equal(obs_mpm.index, exp_mpm.index)
        pd.util.testing.assert_index_equal(obs_mpm.columns, exp_mpm.columns)
        np.testing.assert_allclose(obs_mpm.values, exp_mpm.values, atol=.01)


class PlotHeatmapTests(TestCase):

    def setUp(self):
        vals = np.array([[0., 0.62444444, 0., 0.01555556, 0.36],
                         [0.68444444, 0., 0.09333333, 0.12666667, 0.09555556],
                         [0., 0.00888889, 0., 0.08222222, 0.90888889],
                         [0.19111111, 0.2, 0.5, 0., 0.10888889]])
        source_names = ['source1a', 'source1b', 'source2a', 'source2b']
        self.mpm = pd.DataFrame(vals, index=source_names,
                                columns=source_names + ['Unknown'])

    def test_defaults(self):
        # plot_heatmap call returns successfully
        fig, ax = plot_heatmap(self.mpm)

    def test_non_defaults(self):
        # plot_heatmap call returns successfully
        fig, ax = plot_heatmap(self.mpm, cm=plt.cm.jet,
                               xlabel='Other 1', ylabel='Other 2',
                               title='Other 3')

if __name__ == '__main__':
    main()
