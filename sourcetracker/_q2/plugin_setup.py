#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Biota Technology.
# www.biota.com
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import importlib
import qiime2.plugin
from sourcetracker import __version__
from qiime2.plugin import (Int, Float,
                           Metadata, Str,
                           Bool, Choices)
from q2_types.feature_table import (FeatureTable,
                                    Frequency,
                                    RelativeFrequency)
from q2_types.feature_data import (FeatureData,
                                   Taxonomy)
from sourcetracker._gibbs import gibbs
from sourcetracker._q2._visualizer import (assignment_barplot,
                                           barplot)
# import descriptions
from ._type import SinkSourceMap
from ._format import SinkSourceMapDirectoryFormat
from q2_types.sample_data import SampleData
from sourcetracker._gibbs_defaults import (DESC_TBL, DESC_MAP, DESC_FMAP,
                                           DESC_LOO, DESC_JBS, DESC_ALPH1,
                                           DESC_ALPH2, DESC_BTA, DESC_RAF1,
                                           DESC_RAF2, DESC_RST, DESC_DRW,
                                           DESC_BRN, DESC_DLY, DESC_PFA,
                                           DESC_RPL, DESC_SNK, DESC_SRS,
                                           DESC_SRS2, DESC_CAT, OUT_MEAN,
                                           OUT_STD, OUT_PFA, DESC_PVAL,
                                           OUT_PFAM)

PARAMETERS = {'sample_metadata': Metadata,
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
              'per_sink_feature_assignments': Bool % Choices(True),
              'sample_with_replacement': Bool,
              'source_sink_column': Str,
              'source_column_value': Str,
              'sink_column_value': Str,
              'source_category_column': Str}
PARAMETERDESC = {'sample_metadata': DESC_MAP,
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
                 'per_sink_feature_assignments': 'This feature is disabled' +
                                                 'but is coming soon! ' +
                                                 DESC_PFA,
                 'sample_with_replacement': DESC_RPL,
                 'source_sink_column': DESC_SNK,
                 'source_column_value': DESC_SRS,
                 'sink_column_value': DESC_SRS2,
                 'source_category_column': DESC_CAT}

citations = qiime2.plugin.Citations.load(
    'citations.bib', package='sourcetracker')

plugin = qiime2.plugin.Plugin(
    name='sourcetracker2',
    version=__version__,
    website="https://github.com/biota/sourcetracker2",
    citations=[citations['Knights2011-qx']],
    short_description=('Plugin for source tracking.'),
    description=('This is a QIIME 2 plugin supporting sourcetracker2.'),
    package='sourcetracker')

plugin.methods.register_function(
    function=gibbs,
    inputs={'feature_table': FeatureTable[Frequency]},
    parameters=PARAMETERS,
    outputs=[('mixing_proportions', FeatureTable[RelativeFrequency]),
             ('mixing_proportion_stds', FeatureTable[RelativeFrequency]),
             ('per_sink_assignments', FeatureTable[RelativeFrequency]),
             ('per_sink_assignments_map', SampleData[SinkSourceMap])],
    input_descriptions={'feature_table': DESC_TBL},
    parameter_descriptions=PARAMETERDESC,
    output_descriptions={'mixing_proportions': OUT_MEAN,
                         'mixing_proportion_stds': OUT_STD,
                         'per_sink_assignments': OUT_PFA,
                         'per_sink_assignments_map': OUT_PFAM},
    name='sourcetracker2 gibbs',
    description=('SourceTracker2 is a highly parallel version of '
                 'SourceTracker that was originally described in'
                 ' Knights et al., 2011.'),
)

plugin.visualizers.register_function(
    function=assignment_barplot,
    inputs={'feature_assignments': FeatureTable[RelativeFrequency],
            'feature_metadata': FeatureData[Taxonomy],
            'assignments_map': SampleData[SinkSourceMap]},
    parameters={'per_value': Str},
    input_descriptions={'feature_assignments': OUT_PFA,
                        'feature_metadata': DESC_FMAP,
                        'assignments_map': OUT_PFAM},
    parameter_descriptions={'per_value': DESC_PVAL},
    name='Visualize feature assignments with an interactive bar plot',
    description='This visualizer produces an interactive barplot visualization'
                ' of the feature assignments. '
                'Interactive features include multi-level '
                'sorting, plot recoloring, sample relabeling, and SVG '
                'figure export.'
)

plugin.visualizers.register_function(
    function=barplot,
    inputs={'proportions': FeatureTable[RelativeFrequency]},
    parameters={'sample_metadata': Metadata,
                'category_column': Str},
    input_descriptions={'proportions': OUT_MEAN},
    parameter_descriptions={'sample_metadata': DESC_MAP,
                            'category_column': DESC_CAT},
    name='Visualize feature assignments with an interactive bar plot',
    description='This visualizer produces an interactive barplot visualization'
                ' of the feature assignments. '
                'Interactive features include multi-level '
                'sorting, plot recoloring, sample relabeling, and SVG '
                'figure export.'
)

plugin.register_semantic_types(SinkSourceMap)
plugin.register_semantic_type_to_format(
    SampleData[SinkSourceMap],
    artifact_format=SinkSourceMapDirectoryFormat)
plugin.register_formats(SinkSourceMapDirectoryFormat)
importlib.import_module('sourcetracker._q2._transformer')
