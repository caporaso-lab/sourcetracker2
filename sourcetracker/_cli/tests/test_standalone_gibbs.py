#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import unittest
import tempfile
import pandas as pd
from click.testing import CliRunner
from sourcetracker._cli.gibbs import gibbs
from numpy.testing import assert_allclose


class Test_standalone_gibbs(unittest.TestCase):
    def setUp(self):
        # different cli perams to test
        # all used in the example section
        self.examples = {'example1': {'mapping': 'map.txt',
                                      'restarts': 2,
                                      'draws_per_restart': 3,
                                      'burnin': 10,
                                      'delay': 2,
                                      'loo': False,
                                      'source_sink_column': 'SourceSink',
                                      'source_column_value': 'source',
                                      'sink_column_value': 'sink',
                                      'source_category_column': 'Env',
                                      'sink_rarefaction_depth': 1000,
                                      'source_rarefaction_depth': 1000,
                                      'per_sink_feature_assignments': True},
                         'example2': {'mapping': 'alt-map.txt',
                                      'restarts': 2,
                                      'draws_per_restart': 3,
                                      'burnin': 10,
                                      'delay': 2,
                                      'loo': False,
                                      'source_sink_column': 'source-or-sink',
                                      'source_column_value': 'src',
                                      'sink_column_value': 'snk',
                                      'source_category_column': 'sample-type',
                                      'sink_rarefaction_depth': 1000,
                                      'source_rarefaction_depth': 1000,
                                      'per_sink_feature_assignments': False},
                         'example3': {'mapping': 'map.txt',
                                      'restarts': 2,
                                      'draws_per_restart': 3,
                                      'burnin': 10,
                                      'delay': 2,
                                      'loo': True,
                                      'source_sink_column': 'SourceSink',
                                      'source_column_value': 'source',
                                      'sink_column_value': 'sink',
                                      'source_category_column': 'Env',
                                      'sink_rarefaction_depth': 1000,
                                      'source_rarefaction_depth': 1000,
                                      'per_sink_feature_assignments': False},
                         'example4': {'mapping': 'map.txt',
                                      'restarts': 2,
                                      'draws_per_restart': 3,
                                      'burnin': 25,
                                      'delay': 2,
                                      'loo': False,
                                      'source_sink_column': 'SourceSink',
                                      'source_column_value': 'source',
                                      'sink_column_value': 'sink',
                                      'source_category_column': 'Env',
                                      'sink_rarefaction_depth': 1000,
                                      'source_rarefaction_depth': 1000,
                                      'per_sink_feature_assignments': False},
                         'example5': {'mapping': 'map.txt',
                                      'restarts': 2,
                                      'draws_per_restart': 3,
                                      'burnin': 10,
                                      'delay': 2,
                                      'loo': False,
                                      'source_sink_column': 'SourceSink',
                                      'source_column_value': 'source',
                                      'sink_column_value': 'sink',
                                      'source_category_column': 'Env',
                                      'sink_rarefaction_depth': 1700,
                                      'source_rarefaction_depth': 1500,
                                      'per_sink_feature_assignments': False}}

    def test_standalone_gibbs(self):
        """Checks the output produced by sourcetracker2's standalone script.

           This is more of an "integration test" than a unit test -- the
           details of the algorithm used by the standalone gibbs script are
           checked in more detail in sourctracker/tests.
        """
        crnt_dir = os.path.dirname(os.path.abspath(__file__))
        tst_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir)
        # test the cli for each example peram set
        for exmp_i, perams in self.examples.items():
            # get the tables input pth and out pth
            tbl_pth = os.path.join(tst_pth, 'data/tiny-test/otu_table.biom')
            mta_pth = os.path.join(
                tst_pth, 'data/tiny-test', perams['mapping'])

            # generate the temp. directory to store res
            with tempfile.TemporaryDirectory() as temp_dir_name:
                res_pth = os.path.join(temp_dir_name, 'res_' + exmp_i)

                # add the additional loo flag if needed
                add_ = []
                if perams['loo']:
                    add_ += ['--loo']
                elif perams['per_sink_feature_assignments']:
                    add_ += ['--per_sink_feature_assignments']
                # run the cli using the testing runner
                runner = CliRunner()
                result = runner.invoke(gibbs,
                                       ['--table_fp',
                                        tbl_pth,
                                        '--mapping_fp',
                                        mta_pth,
                                        '--output_dir',
                                        res_pth,
                                        '--restarts',
                                        perams['restarts'],
                                        '--draws_per_restart',
                                        perams['draws_per_restart'],
                                        '--burnin',
                                        perams['burnin'],
                                        '--delay',
                                        perams['delay'],
                                        '--source_sink_column',
                                        perams['source_sink_column'],
                                        '--source_column_value',
                                        perams['source_column_value'],
                                        '--sink_column_value',
                                        perams['sink_column_value'],
                                        '--source_category_column',
                                        perams['source_category_column'],
                                        '--source_category_column',
                                        perams['source_category_column'],
                                        '--sink_rarefaction_depth',
                                        perams['sink_rarefaction_depth'],
                                        '--source_rarefaction_depth',
                                        perams['source_rarefaction_depth']]
                                       + add_)
                # check exit code was 0 (indicating success)
                self.assertEqual(result.exit_code, 0)
                # check mixing proportions are reproduced
                exp_pth = os.path.join(crnt_dir, 'data',
                                       'exp_' + exmp_i,
                                       'mixing_proportions.txt')
                res_pth = os.path.join(temp_dir_name,
                                       'res_' + exmp_i,
                                       'mixing_proportions.txt')
                exp_mp = pd.read_csv(exp_pth, sep='\t', index_col=0).T
                res_mp = pd.read_csv(res_pth, sep='\t', index_col=0)
                # check values
                assert_allclose(exp_mp,
                                res_mp.loc[exp_mp.index,
                                           exp_mp.columns],
                                atol=.50)


if __name__ == "__main__":
    unittest.main()
