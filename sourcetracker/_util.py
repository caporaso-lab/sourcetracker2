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


def parse_sample_metadata(f):
    sample_metadata = pd.read_csv(f, sep='\t', dtype=object)
    sample_metadata.set_index(sample_metadata.columns[0], drop=True,
                              append=False, inplace=True)
    return sample_metadata
