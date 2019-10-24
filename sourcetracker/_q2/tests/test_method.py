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
from biom import Table, load_table
from qiime2 import Artifact
from qiime2 import Metadata
from numpy.testing import assert_allclose
from qiime2.plugins.sourcetracker2.actions import gibbs
from sourcetracker._q2._visualizer import (barplot,
                                           assignment_barplot)


class Test_QIIME2_gibbs(unittest.TestCase):

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
                                      'source_rarefaction_depth': 1000},
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
                                      'source_rarefaction_depth': 1000},
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
                                      'source_rarefaction_depth': 1000},
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
                                      'source_rarefaction_depth': 1000},
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
                                      'source_rarefaction_depth': 1500}}

    def test_q2_gibbs(self):
        """Tests that the Q2 and standalone gibbs results match.

           Also validates against ground truth "expected" results.
        """
        crnt_dir = os.path.dirname(os.path.abspath(__file__))
        tst_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir)
        # test the cli for each example peram set
        for exmp_i, perams in self.examples.items():
            # get the tables input pth and out pth
            tbl_pth = os.path.join(tst_pth, 'data/tiny-test/otu_table.biom')
            tax_pth = os.path.join(tst_pth, 'data/tiny-test/taxonomy.qza')
            mta_pth = os.path.join(
                tst_pth, 'data/tiny-test', perams['mapping'])
            # import tables
            q2table = Artifact.import_data("FeatureTable[Frequency]",
                                           load_table(tbl_pth))
            q2tax = Artifact.load(tax_pth)
            q2meta = Metadata(pd.read_csv(mta_pth,
                                          sep='\t',
                                          index_col=0))
            # Run gemelli through QIIME 2 (specifically, the Artifact API)
            # save a few as var to avoid long lines
            rs_ = perams['source_rarefaction_depth']
            rss_ = perams['sink_rarefaction_depth']
            scv_ = perams['source_column_value']
            scc_ = perams['source_category_column']
            draw_ = perams['draws_per_restart']
            ssc_ = perams['source_sink_column']
            sincv_ = perams['sink_column_value']
            mp, mpstd, fas, fasmf = gibbs(q2table,
                                          q2meta,
                                          loo=perams['loo'],
                                          source_rarefaction_depth=rs_,
                                          sink_rarefaction_depth=rss_,
                                          restarts=perams['restarts'],
                                          draws_per_restart=draw_,
                                          burnin=perams['burnin'],
                                          delay=perams['delay'],
                                          source_sink_column=ssc_,
                                          source_column_value=scv_,
                                          sink_column_value=sincv_,
                                          source_category_column=scc_)
            # run prop barplot
            with tempfile.TemporaryDirectory() as output_dir:
                barplot(output_dir,
                        mp.view(pd.DataFrame),
                        q2meta,
                        scc_)
                index_fp = os.path.join(output_dir, 'index.html')
                self.assertTrue(os.path.exists(index_fp))
            # run a per-sink prop
            if perams['loo']:
                per_ = 'drainwater'
            else:
                per_ = 's0'
            with tempfile.TemporaryDirectory() as output_dir:
                assignment_barplot(output_dir,
                                   fas.view(pd.DataFrame),
                                   q2tax.view(pd.DataFrame),
                                   fasmf.view(pd.DataFrame),
                                   per_)
                index_fp = os.path.join(output_dir, 'index.html')
                self.assertTrue(os.path.exists(index_fp))

            # Get the underlying data from these artifacts
            res_mp = mp.view(Table).to_dataframe().T
            # check mixing proportions from cli
            exp_pth = os.path.join(crnt_dir,
                                   os.pardir,
                                   os.pardir,
                                   '_cli',
                                   'tests',
                                   'data',
                                   'exp_' + exmp_i,
                                   'mixing_proportions.txt')
            exp_mp = pd.read_csv(exp_pth, sep='\t', index_col=0).T
            # compare the results
            assert_allclose(exp_mp,
                            res_mp.loc[exp_mp.index,
                                       exp_mp.columns],
                            atol=.50)


if __name__ == "__main__":
    unittest.main()
