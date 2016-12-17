#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import io
import unittest

from biom.table import Table

import numpy as np

import pandas as pd
import pandas.util.testing as pdt

from sourcetracker._util import parse_sample_metadata, biom_to_df


class ParseSampleMetadata(unittest.TestCase):

    def test_parse_sample_metadata(self):
        map_f = io.StringIO("#SampleID\tCol1\tCol2\n01\ta\t1\n00\tb\t2\n")
        observed = parse_sample_metadata(map_f)
        expected = pd.DataFrame([['a', '1'], ['b', '2']],
                                index=pd.Index(['01', '00'], name='#SampleID'),
                                columns=['Col1', 'Col2'])
        pdt.assert_frame_equal(observed, expected)


class BiomToDF(unittest.TestCase):

    def test_convert(self):
        exp = pd.DataFrame(np.arange(200).reshape(20, 10).astype(np.float64).T,
                           index=['s%s' % i for i in range(10)],
                           columns=['o%s' % i for i in range(20)])

        data = np.arange(200).reshape(20, 10).astype(np.float64)
        oids = ['o%s' % i for i in range(20)]
        sids = ['s%s' % i for i in range(10)]

        obs = biom_to_df(Table(data, oids, sids))
        pd.util.testing.assert_frame_equal(obs, exp)


if __name__ == "__main__":
    unittest.main()
