from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData

SinkSourceMap = SemanticType('SinkSourceMap',
                             variant_of=SampleData.field['type'])
