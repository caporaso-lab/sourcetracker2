import pandas as pd
from qiime2.plugin.testing import TestPluginBase
from sourcetracker._q2._format import SinkSourceMapFormat


class TestSinkSourceMapFormatTransformers(TestPluginBase):
    package = "sourcetracker._q2.tests"

    def test_pd_series_to_transformer(self):

        transformer = self.get_transformer(pd.DataFrame, SinkSourceMapFormat)

        n = 4
        sinks = ['sink%i' % j
                 for j in range(n)
                 for i in range(n)]
        sources = ['source%i' % i
                   for j in range(n)
                   for i in range(n)]
        test_df = pd.DataFrame([sinks, sources],
                               ['Sink', 'Source']).T
        test_df.index = 'sample' + test_df.index.astype(str)
        test_df.index.name = 'sampleid'
        result = transformer(test_df)
        self.assertIsInstance(result, SinkSourceMapFormat)
