#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import pandas as pd
from skbio.util import assert_data_frame_almost_equal

from sourcetracker import compare_sinks, compare_sink_metrics
from sourcetracker._compare import _validate_dataframes


class CompareSinksTests(unittest.TestCase):

    def setUp(self):
        mpm1 = {'Unknown': {'sink1': 0.2829656862745098,
                            'sink2': 0.2700903353545353,
                            'sink3': 0.27551818537348455,
                            'sink4': 0.3444833369795085,
                            'sink5': 0.35963184240271273,
                            'sink6': 0.47133120258770073},
                'source1': {'sink1': 0.281655943627451,
                            'sink2': 0.17903724786536321,
                            'sink3': 0.26814235432147049,
                            'sink4': 0.33563042368555385,
                            'sink5': 0.14895850153398998,
                            'sink6': 0.18439678077708035},
                'source2': {'sink1': 0.28102022058823528,
                            'sink2': 0.27096481458565358,
                            'sink3': 0.1712788423934298,
                            'sink4': 0.17941369503390944,
                            'sink5': 0.36757629581785889,
                            'sink6': 0.17007971042396705},
                'source3': {'sink1': 0.15435814950980392,
                            'sink2': 0.27990760219444788,
                            'sink3': 0.28506061791161519,
                            'sink4': 0.14047254430102823,
                            'sink5': 0.1238333602454384,
                            'sink6': 0.17419230621125187}}
        self.mpm1 = pd.DataFrame(mpm1)

    def test_validate_dataframes_valid(self):
        df1 = {'Unknown': {'sink1': 0.25, 'sink2': 0.25},
               'Source1': {'sink1': 0.50, 'sink2': 0.25},
               'Source2': {'sink1': 0.25, 'sink2': 0.25}}
        df1 = pd.DataFrame(df1)
        df2 = {'Unknown': {'sink1': 0.1, 'sink2': 0.25},
               'Source1': {'sink1': 0.1, 'sink2': 0.25},
               'Source2': {'sink1': 0.8, 'sink2': 0.25}}
        df2 = pd.DataFrame(df2)
        _validate_dataframes(df1, df2)

    def test_validate_dataframes_invalid_sources(self):
        df1 = {'Hello': {'sink1': 0.25, 'sink2': 0.25},
               'Source1': {'sink1': 0.50, 'sink2': 0.25},
               'Source2': {'sink1': 0.25, 'sink2': 0.25}}
        df1 = pd.DataFrame(df1)
        df2 = {'Unknown': {'sink1': 0.1, 'sink2': 0.25},
               'Source1': {'sink1': 0.1, 'sink2': 0.25},
               'Source2': {'sink1': 0.8, 'sink2': 0.25}}
        df2 = pd.DataFrame(df2)
        with self.assertRaises(ValueError):
            _validate_dataframes(df1, df2)

        df1 = {'Source1': {'sink1': 0.50, 'sink2': 0.25},
               'Source2': {'sink1': 0.50, 'sink2': 0.25}}
        df1 = pd.DataFrame(df1)
        df2 = {'Unknown': {'sink1': 0.1, 'sink2': 0.25},
               'Source1': {'sink1': 0.1, 'sink2': 0.25},
               'Source2': {'sink1': 0.8, 'sink2': 0.25}}
        df2 = pd.DataFrame(df2)
        with self.assertRaises(ValueError):
            _validate_dataframes(df1, df2)

    def test_validate_dataframes_invalid_sinks(self):
        df1 = {'Unknown': {'sink1': 0.25, 'sink3': 0.25},
               'Source1': {'sink1': 0.50, 'sink3': 0.25},
               'Source2': {'sink1': 0.25, 'sink3': 0.25}}
        df1 = pd.DataFrame(df1)
        df2 = {'Unknown': {'sink1': 0.1, 'sink2': 0.25},
               'Source1': {'sink1': 0.1, 'sink2': 0.25},
               'Source2': {'sink1': 0.8, 'sink2': 0.25}}
        df2 = pd.DataFrame(df2)
        with self.assertRaises(ValueError):
            _validate_dataframes(df1, df2)

        df1 = {'Unknown': {'sink1': 0.25},
               'Source1': {'sink1': 0.50},
               'Source2': {'sink1': 0.50}}
        df1 = pd.DataFrame(df1)
        df2 = {'Unknown': {'sink1': 0.1, 'sink2': 0.25},
               'Source1': {'sink1': 0.1, 'sink2': 0.25},
               'Source2': {'sink1': 0.8, 'sink2': 0.25}}
        df2 = pd.DataFrame(df2)
        with self.assertRaises(ValueError):
            _validate_dataframes(df1, df2)

    def test_compare_sink_metrics(self):
        self.assertEqual(compare_sink_metrics(),
                         ['absolute_difference', 'euclidean', 'pearson',
                          'spearman'])

    def test_unknown_metric(self):
        with self.assertRaises(KeyError):
            compare_sinks(self.mpm1, self.mpm1, 'not-a-metric')

    def test_non_overlapping_sinks(self):
        mpm2 = self.mpm1.copy()
        mpm2.index = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink7']
        with self.assertRaisesRegex(ValueError, 'Sinks'):
            compare_sinks(self.mpm1, mpm2, 'spearman')

    def test_non_overlapping_sources(self):
        mpm2 = self.mpm1.copy()
        mpm2.columns = ['source1', 'source2', 'source4', 'Unknown']
        with self.assertRaisesRegex(ValueError, 'Sources'):
            compare_sinks(self.mpm1, mpm2, 'spearman')

    def test_order_independence_sinks(self):
        mpm1 = self.mpm1.sort_index(ascending=False)
        mpm2 = self.mpm1.copy().sort_index(ascending=True)
        # confirm that the indices are now different
        self.assertEqual(list(mpm1.index), list(reversed(mpm2.index)))

        observed = compare_sinks(self.mpm1, self.mpm1, 'spearman')
        expected_ids = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink6']
        expected_values = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                           (1.0, 0.0), (1.0, 0.0)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Spearman rho', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_order_independence_sources(self):
        mpm1 = self.mpm1.sort_index(ascending=False, axis=1)
        mpm2 = self.mpm1.copy().sort_index(ascending=True, axis=1)
        # confirm that the columns are now different
        self.assertEqual(list(mpm1.columns), list(reversed(mpm2.columns)))

        observed = compare_sinks(self.mpm1, self.mpm1, 'spearman')
        expected_ids = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink6']
        expected_values = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                           (1.0, 0.0), (1.0, 0.0)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Spearman rho', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_spearman_perfect(self):
        observed = compare_sinks(self.mpm1, self.mpm1, 'spearman')
        expected_ids = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink6']
        expected_values = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                           (1.0, 0.0), (1.0, 0.0)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Spearman rho', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_spearman(self):
        mpm1 = {'Unknown': {'sink1': 0.25},
                'Source1': {'sink1': 0.50},
                'Source2': {'sink1': 0.25}}
        mpm1 = pd.DataFrame(mpm1)
        mpm2 = {'Unknown': {'sink1': 0.1},
                'Source1': {'sink1': 0.1},
                'Source2': {'sink1': 0.8}}
        mpm2 = pd.DataFrame(mpm2)

        observed = compare_sinks(mpm1, mpm2, 'spearman')
        expected_ids = ['sink1']
        # expected values computed by calling scipy.stats.spearmanr directly
        expected_values = [(-0.5, 2./3)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Spearman rho', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_pearson_perfect(self):
        observed = compare_sinks(self.mpm1, self.mpm1, 'pearson')
        expected_ids = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink6']
        expected_values = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0),
                           (1.0, 0.0), (1.0, 0.0)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Pearson r', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_pearsonr(self):
        mpm1 = {'Unknown': {'sink1': 0.25},
                'Source1': {'sink1': 0.50},
                'Source2': {'sink1': 0.25}}
        mpm1 = pd.DataFrame(mpm1)
        mpm2 = {'Unknown': {'sink1': 0.1},
                'Source1': {'sink1': 0.1},
                'Source2': {'sink1': 0.8}}
        mpm2 = pd.DataFrame(mpm2)

        observed = compare_sinks(mpm1, mpm2, 'pearson')
        expected_ids = ['sink1']
        # expected values computed by calling scipy.stats.pearsonr directly
        expected_values = [(-0.5, 2./3)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Pearson r', 'p'])
        assert_data_frame_almost_equal(observed, expected)

    def test_euclidean_perfect(self):
        observed = compare_sinks(self.mpm1, self.mpm1, 'euclidean')
        expected_ids = ['sink1', 'sink2', 'sink3', 'sink4', 'sink5', 'sink6']
        expected_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Euclidean distance'])
        assert_data_frame_almost_equal(observed, expected)

    def test_euclidean(self):
        mpm1 = {'Unknown': {'sink1': 0.25},
                'Source1': {'sink1': 0.50},
                'Source2': {'sink1': 0.25}}
        mpm1 = pd.DataFrame(mpm1)
        mpm2 = {'Unknown': {'sink1': 0.1},
                'Source1': {'sink1': 0.1},
                'Source2': {'sink1': 0.8}}
        mpm2 = pd.DataFrame(mpm2)

        observed = compare_sinks(mpm1, mpm2, 'euclidean')
        expected_ids = ['sink1']
        # expected values computed by calling
        # scipy.stats.spatial.distance.euclidean directly
        expected_values = [0.6964194]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Euclidean distance'])
        assert_data_frame_almost_equal(observed, expected)

    def test_absolute_difference(self):
        mpm1 = {'Unknown': {'sink1': 0.25},
                'Source1': {'sink1': 0.50},
                'Source2': {'sink1': 0.25}}
        mpm1 = pd.DataFrame(mpm1)
        mpm2 = {'Unknown': {'sink1': 0.1},
                'Source2': {'sink1': 0.8},
                'Source1': {'sink1': 0.1}}
        mpm2 = pd.DataFrame(mpm2)

        observed = compare_sinks(mpm1, mpm2, 'absolute_difference')
        expected_ids = ['sink1']
        # expected values computed by hand
        expected_values = [(0.4, 0.55, 0.15)]
        expected = pd.DataFrame(expected_values, index=expected_ids,
                                columns=['Source1', 'Source2', 'Unknown'])
        assert_data_frame_almost_equal(observed, expected)
