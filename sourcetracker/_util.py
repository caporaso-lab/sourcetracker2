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
    """Parse QIIME 1-like sample metadata file.

    Parameters
    ----------
    f : file handle
        The sample metadata to be parsed.

    Returns
    -------
    pd.DataFrame
        Sample metadata where index is sample ids and columns are metadata
        categories.

    """
    sample_metadata = pd.read_csv(f, sep='\t', dtype=object)
    sample_metadata.set_index(sample_metadata.columns[0], drop=True,
                              append=False, inplace=True)
    return sample_metadata


def biom_to_df(biom_table):
    '''Turn biom table into dataframe.

    Parameters
    ----------
    biom_table : biom.table.Table
        Biom table.

    Returns
    -------
    feature_table : pd.DataFrame
        Contingency table with rows, columns = samples, features.
    '''
    return pd.DataFrame(biom_table._data.toarray().T,
                        index=biom_table.ids(axis='sample'),
                        columns=biom_table.ids(axis='observation'))
