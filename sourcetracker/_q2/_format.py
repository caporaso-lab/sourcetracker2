import qiime2.plugin.model as model
from qiime2.plugin import ValidationError


class SinkSourceMapFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # check the header column names
            header = fh.readline()
            comp_columns = list(header.split('\t'))[1:]
            # ensure both headers are present
            allowed_ = ['Source', 'Sink', 'Source_one', 'Source_two']
            num_col = sum([str(i).replace('\n', '') in allowed_
                           for i in comp_columns])
            if num_col != 2:
                raise ValidationError('Source or Sink columns are missing.'
                                      ' Got %s' % ', '.join(comp_columns))
            # validate the body of the data
            for line_number, line in enumerate(fh, start=2):
                cells = line.split('\t')
                values_ = [is_str(cells[c].strip()) for c in [1, 2]]
                if not all(values_):
                    err_ = 'Non string values in source-sink map.'
                    raise ValidationError(err_)
                if n_records is not None and (line_number - 1) >= n_records:
                    break

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


def is_str(val):
    try:
        str(val)
        return True
    except ValueError:
        return False


SinkSourceMapDirectoryFormat = model.SingleFileDirectoryFormat(
    'SinkSourceMapDirectoryFormat', 'SinkSourceMap.tsv',
    SinkSourceMapFormat)
