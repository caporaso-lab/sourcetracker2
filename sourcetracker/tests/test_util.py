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

import pandas as pd
import pandas.util.testing as pdt

from sourcetracker._util import parse_sample_metadata


class ParseSampleMetadata(unittest.TestCase):

    def test_parse_sample_metadata(self):
        map_f = io.StringIO("#SampleID\tCol1\tCol2\n01\ta\t1\n00\tb\t2\n")
        observed = parse_sample_metadata(map_f)
        expected = pd.DataFrame([['a', '1'], ['b', '2']],
                                index=pd.Index(['01', '00'], name='#SampleID'),
                                columns=['Col1', 'Col2'])
        pdt.assert_frame_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
