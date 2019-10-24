from skbio.util import get_data_path
from qiime2.plugin import ValidationError
from qiime2.plugin.testing import TestPluginBase
from sourcetracker._q2._format import SinkSourceMapFormat


class TestSinkSourceMapFormatTest(TestPluginBase):
    package = "sourcetracker._q2.tests"

    def test_valid_simple(self):
        filepath = get_data_path('SinkSourceMap.tsv')
        format = SinkSourceMapFormat(filepath, mode='r')

        format.validate('min')
        format.validate('max')

    def test_valid_real_data(self):
        filepath = get_data_path('SinkSourceMap_noloo.tsv')
        format = SinkSourceMapFormat(filepath, mode='r')

        format.validate('min')
        format.validate('max')

    def test_invalid_header(self):
        filepath = get_data_path('SinkSourceMap-invalid-header.tsv')
        format = SinkSourceMapFormat(filepath, mode='r')
        for level in 'min', 'max':
            with self.assertRaises(ValidationError):
                format.validate(level)
