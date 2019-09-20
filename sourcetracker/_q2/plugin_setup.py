#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2.plugin
from sourcetracker import __version__
from qiime2.plugin import (Properties, Int, Float,
                           Metadata, Str, Bool, Choices)
from q2_types.feature_table import (FeatureTable, Frequency,
                                    RelativeFrequency)
from sourcetracker._gibbs import gibbs_helper as gibbs

# import default descriptions
from sourcetracker._gibbs_defaults import (DESC_TBL, DESC_MAP, DESC_OUT,
                                           DESC_LOO, DESC_JBS, DESC_ALPH1,
                                           DESC_ALPH2, DESC_BTA, DESC_RAF1,
                                           DESC_RAF2, DESC_RST, DESC_DRW,
                                           DESC_BRN, DESC_DLY, DESC_PFA,
                                           DESC_RPL, DESC_SNK, DESC_SRS,
                                           DESC_SRS2, DESC_CAT, OUT_MEAN,
                                           OUT_STD)

# import default values
from sourcetracker._gibbs_defaults import (DEFAULT_ALPH1, DEFAULT_ALPH2,
                                           DEFAULT_TEN, DEFAULT_ONE,
                                           DEFAULT_HUND, DEFAULT_THOUS,
                                           DEFAULT_FLS, DEFAULT_SNK,
                                           DEFAULT_SRS, DEFAULT_SRS2,
                                           DEFAULT_CAT)

PARAMETERS = {'mapping_fp': Metadata,
              'loo': Bool,
              'jobs': Int,
              'alpha1': Float,
              'alpha2': Float,
              'beta': Float,
              'source_rarefaction_depth': Int,
              'sink_rarefaction_depth': Int,
              'restarts': Int,
              'draws_per_restart': Int,
              'burnin': Int,
              'delay': Int,
              'per_sink_feature_assignments': Bool % Choices(False),
              'sample_with_replacement': Bool,
              'source_sink_column': Str,
              'source_column_value': Str,
              'sink_column_value': Str,
              'source_category_column': Str}
PARAMETERDESC = {'mapping_fp': DESC_MAP,
                 'loo': DESC_LOO,
                 'jobs': DESC_JBS,
                 'alpha1': DESC_ALPH1,
                 'alpha2': DESC_ALPH2,
                 'beta': DESC_BTA,
                 'source_rarefaction_depth': DESC_RAF1,
                 'sink_rarefaction_depth': DESC_RAF2,
                 'restarts': DESC_RST,
                 'draws_per_restart': DESC_DRW,
                 'burnin': DESC_BRN,
                 'delay': DESC_DLY,
                 'per_sink_feature_assignments': DESC_PFA,
                 'sample_with_replacement': DESC_RPL,
                 'source_sink_column': DESC_SNK,
                 'source_column_value': DESC_SRS,
                 'sink_column_value': DESC_SRS2,
                 'source_category_column': DESC_CAT}

citations = qiime2.plugin.Citations.load(
    'citations.bib', package='sourcetracker2')

plugin = qiime2.plugin.Plugin(
    name='sourcetracker2',
    version=__version__,
    website="https://github.com/biota/sourcetracker2",
    citations=[citations['Knights2011-qx']],
    short_description=('Plugin for source tracking.'),
    description=('This is a QIIME 2 plugin supporting sourcetracker2.'),
    package='sourcetracker2')

plugin.methods.register_function(
    function=gibbs,
    inputs={'table_fp': FeatureTable[Frequency]},
    parameters=PARAMETERS,
    outputs=[('mixing_proporitions', FeatureTable[RelativeFrequency]),
             ('mixing_proportion_stds', FeatureTable[RelativeFrequency])],
    input_descriptions={'table': DESC_TBL},
    parameter_descriptions=PARAMETERDESC,
    output_descriptions={'mixing_proporitions': OUT_MEAN,
                         'mixing_proportion_stds': OUT_STD},
    name='sourcetracker2 gibbs',
    description=('SourceTracker2 is a highly parallel version of '
                 'SourceTracker that was originally described in'
                 ' Knights et al., 2011.'),
)