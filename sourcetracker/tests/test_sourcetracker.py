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
from biom.table import Table
from sourcetracker.sourcetracker import (parse_mapping_file, collapse_sources,
                                         ConditionalProbability, gibbs_sampler,
                                         Sampler, sinks_and_sources,
                                         _cli_sync_biom_and_sample_metadata,
                                         _cli_single_sample_formatter,
                                         _cli_collate_results,
                                         subsample_sources_sinks)


class TestPreparationFunctions(TestCase):

    def test_parse_mapping_file(self):
        lines = ['#SampleID\tcat1\tcat2',
                 'S1\t1\ta',
                 'S2\t2\tb',
                 'S3\tsdsd    \tc',
                 'S4\t1111\t9']
        exp = {'S1': {'cat1': '1', 'cat2': 'a'},
               'S2': {'cat1': '2', 'cat2': 'b'},
               'S3': {'cat1': 'sdsd', 'cat2': 'c'},
               'S4': {'cat1': '1111', 'cat2': '9'}}
        obs = parse_mapping_file(lines)
        self.assertEqual(obs, exp)

    def test_collapse_sources(self):
        # The order of the collapsed data is unclear.
        data = np.arange(50).reshape(10, 5)
        oids = ['o%s' % i for i in range(10)]
        sids = ['s%s' % i for i in range(5)]
        biom_table = Table(data, oids, sids)
        sample_metadata = {'s4': {'cat1': '1', 'cat2': 'D'},
                           's0': {'cat1': '1', 'cat2': 'B'},
                           's1': {'cat1': '2', 'cat2': 'C'},
                           's3': {'cat1': '2', 'cat2': 'A'},
                           's2': {'cat1': '2', 'cat2': 'D'}}

        category = 'cat1'
        samples = ['s0', 's1']
        sort = True
        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        exp_envs = np.array(['1', '2'])
        exp_collapsed_sources = \
            np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                      [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]])

        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

        # Change the order of the samples, sort being true should return the
        # same data.
        samples = ['s1', 's0']
        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

        category = 'cat1'
        samples = ['s2', 's0', 's1']
        sort = True
        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        exp_envs = np.array(['1', '2'])
        exp_collapsed_sources = \
            np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                      [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]])
        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

        category = 'cat1'
        samples = ['s4', 's2', 's0', 's1']
        sort = True
        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        exp_envs = np.array(['1', '2'])
        exp_collapsed_sources = \
            np.array([[4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
                      [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]])
        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

        category = 'cat2'
        samples = ['s4', 's2', 's0', 's1']
        sort = True
        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        exp_envs = np.array(['B', 'C', 'D'])
        exp_collapsed_sources = \
            np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                      [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
                      [6, 16, 26, 36, 46, 56, 66, 76, 86, 96]])
        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

        data = np.arange(200).reshape(20, 10)
        oids = ['o%s' % i for i in range(20)]
        sids = ['s%s' % i for i in range(10)]
        biom_table = Table(data, oids, sids)
        sample_metadata = \
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

        category = 'cat4'
        samples = ['s4', 's9', 's0', 's2']
        sort = False

        obs_envs, obs_collapsed_sources = collapse_sources(samples,
                                                           sample_metadata,
                                                           category,
                                                           biom_table, sort)
        exp_envs = np.array(['D', '0'])
        exp_collapsed_sources = \
            np.array([[6, 36, 66, 96, 126, 156, 186, 216, 246, 276, 306, 336,
                       366, 396, 426, 456, 486, 516, 546, 576],
                      [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129,
                       139, 149, 159, 169, 179, 189, 199]])
        np.testing.assert_array_equal(obs_collapsed_sources,
                                      exp_collapsed_sources)
        np.testing.assert_array_equal(obs_envs, exp_envs)

    def test_subsample_sources_sinks(self):
        sources_data = np.array([[5, 100, 3, 0, 0, 1, 9],
                                 [2, 20, 1, 0, 0, 0, 98],
                                 [1000, 0, 0, 0, 0, 0, 0]])
        sinks_data = np.array([[200, 0, 11, 400, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 50]])
        sinks = np.array(['sink1', 'sink2'])
        sources = np.array(['source1', 'source2', 'source3'])
        # The table is composed of the 3 sources and 2 sinks. We concatenate
        # the sink and source data together to create this table
        feature_table = Table(np.vstack((sources_data, sinks_data)).T,
                              ['o%s' % i for i in range(7)],
                              np.hstack((sources, sinks)))

        # Test that errors are thrown appropriately.
        sources_depth = 1001
        sinks_depth = 0
        self.assertRaises(ValueError, subsample_sources_sinks, sources_data,
                          sinks, feature_table, sources_depth, sinks_depth)
        sources_depth = 100
        sinks_depth = 51
        self.assertRaises(ValueError, subsample_sources_sinks, sources_data,
                          sinks, feature_table, sources_depth, sinks_depth)

        # Test when no rarefaction would occur.
        sources_depth = 0
        sinks_depth = 0
        obs_rsd, obs_rft = subsample_sources_sinks(sources_data, sinks,
                                                   feature_table,
                                                   sources_depth, sinks_depth)
        np.testing.assert_array_equal(obs_rsd, sources_data)
        self.assertEqual(obs_rft, feature_table)

        # Test with only sources rarefaction.
        # This won't work since cython is generating the PRNG calls, instead,
        # we can settle for less - ensure that the sums are correct.
        # np.random.seed(0)
        # sources_depth = 100
        # sinks_depth = 0
        # obs_rsd, obs_rft = subsample_sources_sinks(sources_data, sinks,
        #                                            feature_table,
        #                                            sources_depth,
        #                                            sinks_depth)
        # exp_rsd = np.array([[5., 84., 2., 0., 0., 1., 8.],
        #                     [1., 16., 1., 0., 0., 0., 82.],
        #                     [100., 0., 0., 0., 0., 0., 0.]])
        # np.testing.assert_array_equal(obs_rsd, sources_data)
        # self.assertEqual(obs_rft, feature_table)
        sources_depth = 100
        sinks_depth = 0
        obs_rsd, obs_rft = subsample_sources_sinks(sources_data, sinks,
                                                   feature_table,
                                                   sources_depth, sinks_depth)
        np.testing.assert_array_equal(obs_rsd.sum(1), np.array([100]*3))
        self.assertEqual(obs_rft, feature_table)

        # Test with only sinks rarefaction.
        # This won't work since cython is generating the PRNG calls, instead,
        # we can settle for less - ensure that the sums are correct.
        # np.random.seed(0)
        # sources_depth = 0
        # sinks_depth = 49
        # obs_rsd, obs_rft = subsample_sources_sinks(sources_data, sinks,
        #                                            feature_table,
        #                                            sources_depth,
        #                                            sinks_depth)
        # exp_rft_array = np.array([[5., 2., 1000.,  11., 0.],
        #                          [100., 20., 0., 0., 0.],
        #                          [3., 1., 0., 3., 0.],
        #                          [0., 0., 0., 35., 0.],
        #                          [0., 0., 0., 0., 0.],
        #                          [1., 0., 0., 0., 0.],
        #                          [9., 98., 0., 0., 49.]])
        # exp_rft_oids = feature_table.ids(axis='observation')
        # exp_rft_sids = feature_table.ids(axis='sample')
        # exp_rft = Table(exp_rft_array, exp_rft_oids, exp_rft_sids)
        # np.testing.assert_array_equal(obs_rsd, sources_data)
        # self.assertEqual(obs_rft, exp_rft)
        sources_depth = 0
        sinks_depth = 49
        obs_rsd, obs_rft = subsample_sources_sinks(sources_data, sinks,
                                                   feature_table,
                                                   sources_depth, sinks_depth)
        fft = obs_rft.filter(sinks, inplace=False)
        np.testing.assert_array_equal(fft._data.toarray().sum(0),
                                      np.array([49]*2))
        np.testing.assert_array_equal(obs_rsd, sources_data)


class TestCLIFunctions(TestCase):
    '''Tests for the functions which convert command line options.'''
    def setUp(self):
        self.sample_metadata_1 = \
            {'s1': {'source_sink': 'source1', 'cat2': 'random_nonsense'},
             's2': {'source_sink': 'sink', 'cat2': 'sink'},
             's5': {'source_sink': 'source1', 'cat2': 'random_nonsense'},
             's0': {'source_sink': 'source2', 'cat2': 'random_nonsense'},
             's100': {'source_sink': 'sink', 'cat2': 'sink'}}
        # Data for testing sinks_and_sources
        self.sample_metadata_2 = \
            {'s1': {'SourceSink': 'source', 'Env': 'source1'},
             's2': {'SourceSink': 'sink', 'Env': 'e1'},
             's5': {'SourceSink': 'source', 'Env': 'source1'},
             's0': {'SourceSink': 'source', 'Env': 'source2'},
             's100': {'SourceSink': 'sink', 'Env': 'e2'}}
        self.sample_metadata_3 = \
            {'s1': {'SourceSink': 'source', 'Env': 'source1'},
             's2': {'SourceSink': 'source', 'Env': 'e1'},
             's5': {'SourceSink': 'source', 'Env': 'source1'},
             's0': {'SourceSink': 'source', 'Env': 'source2'},
             's100': {'SourceSink': 'source', 'Env': 'e2'}}
        # Data for testing _cli_sync_biom_and_sample_metadata
        oids = ['o1', 'o2', 'o3']
        # Data for an example where samples are removed from biom table only.
        sids = ['Sample1', 'Sample2', 'Sample3', 'Sample4']
        bt_1_data = np.arange(12).reshape(3, 4)
        self.bt_1_in = Table(bt_1_data, oids, sids)
        self.bt_1_out = Table(bt_1_data[:, :-1], oids, sids[:-1])
        self.mf_1_in = \
            {'Sample1': {'cat1': 'X', 'cat2': 'Y'},
             'Sample2': {'cat1': 'X', 'cat2': 'Y'},
             'Sample3': {'cat1': 'X', 'cat2': 'Y'}}
        self.mf_1_out = self.mf_1_in
        # Data for an example where sample are removed from mapping file only.
        self.bt_2_in = self.bt_1_in
        self.bt_2_out = self.bt_1_in
        self.mf_2_in = \
            {'Sample1': {'cat1': 'X', 'cat2': 'Y'},
             'Sample6': {'cat1': 'X', 'cat2': 'Y'},
             'Sample3': {'cat1': 'X', 'cat2': 'Y'},
             'Sample4': {'cat1': 'X', 'cat2': 'Y'},
             'Sample2': {'cat1': 'X', 'cat2': 'Y'}}
        self.mf_2_out = \
            {'Sample1': {'cat1': 'X', 'cat2': 'Y'},
             'Sample3': {'cat1': 'X', 'cat2': 'Y'},
             'Sample4': {'cat1': 'X', 'cat2': 'Y'},
             'Sample2': {'cat1': 'X', 'cat2': 'Y'}}
        # Data for an example where samples are removed from mapping file and
        # biom file.
        sids = ['Sample1', 'sampleA', 'Sample3', 'Sample4']
        bt_3_data = np.arange(12).reshape(3, 4)
        self.bt_3_in = Table(bt_3_data, oids, sids)
        self.bt_3_out = Table(bt_1_data[:, [0, 2, 3]], oids,
                              [sids[0], sids[2], sids[3]])
        self.mf_3_in = self.mf_2_out
        self.mf_3_out = \
            {'Sample1': {'cat1': 'X', 'cat2': 'Y'},
             'Sample3': {'cat1': 'X', 'cat2': 'Y'},
             'Sample4': {'cat1': 'X', 'cat2': 'Y'}}

    def test_sinks_and_sources(self):
        # Test single category, sink, and source identifier example.
        obs_source_samples, obs_sink_samples = \
            sinks_and_sources(self.sample_metadata_2)
        exp_source_samples = ['s1', 's5', 's0']
        exp_sink_samples = ['s2', 's100']
        self.assertEqual(set(exp_sink_samples), set(obs_sink_samples))
        self.assertEqual(set(exp_source_samples), set(obs_source_samples))

        obs_source_samples, obs_sink_samples = \
            sinks_and_sources(self.sample_metadata_3)
        exp_source_samples = ['s1', 's5', 's0', 's2', 's100']
        exp_sink_samples = []
        self.assertEqual(set(exp_sink_samples), set(obs_sink_samples))
        self.assertEqual(set(exp_source_samples), set(obs_source_samples))

    def test__cli_sync_biom_and_sample_metadata(self):
        # Test when syncing removes from mapping file only.
        mf_obs, bt_obs = _cli_sync_biom_and_sample_metadata(self.mf_1_in,
                                                            self.bt_1_in)
        self.assertEqual(mf_obs, self.mf_1_out)
        self.assertEqual(bt_obs, self.bt_1_out)

        # Test when syncing removes from biom table only.
        mf_obs, bt_obs = _cli_sync_biom_and_sample_metadata(self.mf_2_in,
                                                            self.bt_2_in)
        self.assertEqual(mf_obs, self.mf_2_out)
        self.assertEqual(bt_obs, self.bt_2_out)

        # Test when syncing removes from both mapping and biom files.
        mf_obs, bt_obs = _cli_sync_biom_and_sample_metadata(self.mf_3_in,
                                                            self.bt_3_in)
        self.assertEqual(mf_obs, self.mf_3_out)
        self.assertEqual(bt_obs, self.bt_3_out)

        # Test that a ValueError is raised when no samples are shared.
        self.assertRaises(ValueError, _cli_sync_biom_and_sample_metadata,
                          self.sample_metadata_1, self.bt_1_in)

    def test__cli_single_sample_formatter(self):
        proportions = np.arange(20, dtype=int).reshape(5, 4)
        obs = _cli_single_sample_formatter(proportions)
        exp = ('0\t1\t2\t3\n4\t5\t6\t7\n8\t9\t10\t11\n12\t13\t14\t15\n16\t17'
               '\t18\t19')
        self.assertEqual(obs, exp)

    def test__cli_collate_results(self):
        samples = ['s1', 's2', 'sC12', 's4']
        samples_data = [np.arange(18).reshape(6, 3),
                        4 * np.arange(18).reshape(6, 3),
                        200 + np.arange(18).reshape(6, 3),
                        5 + 1000 * np.arange(18).reshape(6, 3)]
        env_ids = np.array(['e1', 'asdf'])
        obs_means, obs_stds = _cli_collate_results(samples, samples_data,
                                                   env_ids)
        exp_means = '\n'.join(['SampleID\te1\tasdf\tUnknown',
                               's1\t7.5\t8.5\t9.5',
                               's2\t30.0\t34.0\t38.0',
                               'sC12\t207.5\t208.5\t209.5',
                               's4\t7505.0\t8505.0\t9505.0'])
        exp_stds = '\n'.join([
            'SampleID\te1\tasdf\tUnknown',
            's1\t5.12347538298\t5.12347538298\t5.12347538298',
            's2\t20.4939015319\t20.4939015319\t20.4939015319',
            'sC12\t5.12347538298\t5.12347538298\t5.12347538298',
            's4\t5123.47538298\t5123.47538298\t5123.47538298'])
        self.assertEqual(obs_means, exp_means)
        self.assertEqual(obs_stds, exp_stds)


class TestSamplerClass(TestCase):
    '''Unit tests for the Python SourceTracker `Sampler` class.'''

    def setUp(self):
        self.sampler_data = np.array([4., 5., 6.])
        self.num_sources = 3
        self.sum = 15  # number of seqs in the sink
        self.sampler = Sampler(self.sampler_data, self.num_sources)

    def test_generate_taxon_sequence(self):
        exp_taxon_sequence = np.array([0., 0., 0., 0., 1., 1., 1., 1., 1., 2.,
                                       2., 2., 2., 2., 2.])

        self.sampler.generate_taxon_sequence()
        obs_taxon_sequence = self.sampler.taxon_sequence

        np.testing.assert_array_equal(obs_taxon_sequence, exp_taxon_sequence)

    def test_generate_environment_assignment(self):
        np.random.seed(0)
        self.sampler.generate_environment_assignments()

        exp_seq_env_assignments = np.array([0, 1, 0, 1, 1, 2, 0, 2, 0, 0, 0, 2,
                                            1, 2, 2])
        obs_seq_env_assignemnts = self.sampler.seq_env_assignments
        np.testing.assert_array_equal(obs_seq_env_assignemnts,
                                      exp_seq_env_assignments)

        exp_envcounts = np.array([6, 4, 5])
        obs_envcounts = self.sampler.envcounts
        np.testing.assert_array_equal(obs_envcounts, exp_envcounts)

    def test_seq_assignments_to_contingency_table(self):
        np.random.seed(0)
        self.sampler.generate_taxon_sequence()
        self.sampler.generate_environment_assignments()
        obs_ct = self.sampler.seq_assignments_to_contingency_table()

        exp_ct = np.array([[2., 2., 0.],
                           [2., 1., 2.],
                           [2., 1., 3.]])

        np.testing.assert_array_equal(obs_ct, exp_ct)


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


class TestGibbsDeterministic(TestCase):
    '''Unit tests for Gibbs based on seeding the PRNG and hand calculations.'''

    def test_single_pass(self):
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
        obs_mps, obs_ct = gibbs_sampler(cp, sink, restarts, draws_per_restart,
                                        burnin, delay)

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

        prps = envcounts / float(envcounts.sum())
        exp_mps = prps/prps.sum()
        # Create taxon table like Sampler class would.
        exp_ct = np.zeros((4, 3))
        for i in range(9):
            exp_ct[expected_et_pairs[1, i],
                   np.int(seq_env_assignments[i])] += 1

        np.testing.assert_array_almost_equal(obs_mps.squeeze(), exp_mps)
        np.testing.assert_array_equal(obs_ct.squeeze(), exp_ct)

if __name__ == '__main__':
    main()
