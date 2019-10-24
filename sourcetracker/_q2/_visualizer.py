import os
import pandas as pd
from qiime2 import Metadata
from q2_taxa._visualizer import barplot as _barplot
from sourcetracker._gibbs_defaults import (DEFAULT_CAT)


def barplot(output_dir: str,
            proportions: pd.DataFrame,
            sample_metadata: Metadata,
            category_column: str = DEFAULT_CAT) -> None:

    # scriptable metadata
    sample_metadata = sample_metadata.to_dataframe()

    # make the sample metadata
    # check if proportion index in metadata index
    if sum([i in sample_metadata.index
            for i in proportions.columns]) > 0:
        # then subset sample metadata by index
        mf_samples = sample_metadata.loc[proportions.columns, :]
        mf_samples.index.name = 'sampleid'
    else:
        # else subset sample metadata by category (in loo case)
        keep_ = sample_metadata[category_column].isin(proportions.columns)
        mf_samples = sample_metadata[keep_]
        mf_samples = mf_samples.set_index(category_column)
        mf_samples = mf_samples.loc[~mf_samples.index.duplicated(keep='first')]
        mf_samples[category_column] = list(mf_samples.index)
        mf_samples = mf_samples[mf_samples.columns[::-1]]
        mf_samples.index.name = 'sampleid'

    # make the feature metadata (mock taxonomy)
    keep_ = sample_metadata[category_column].isin(proportions.index)
    mf_feature = sample_metadata[keep_]
    mf_feature = mf_feature.set_index(category_column)
    mf_feature = mf_feature.loc[~mf_feature.index.duplicated(keep='first')]
    mf_feature.loc['Unknown', :] = 'Unknown'
    mf_feature[category_column] = list(mf_feature.index)
    mf_feature = mf_feature[mf_feature.columns[::-1]]
    mf_feature = mf_feature.astype(str).apply(lambda x: '; '.join(x), axis=1)
    mf_feature = pd.DataFrame(mf_feature,
                              columns=['Taxon'])
    mf_feature.index.name = 'Feature ID'

    # make barplot
    _barplot(output_dir,
             proportions.T,
             pd.Series(mf_feature.Taxon),
             Metadata(mf_samples))

    # grab bundle location to fix
    bundle = os.path.join(output_dir,
                          'dist',
                          'bundle.js')
    # bundle terms to fix for our purpose
    bundle_rplc = {'Relative Frequency': 'Source Contribution',
                   'Taxonomic Level': 'Source Grouping',
                   'Sample': 'Sink'}
    # make small text chnage to bundle
    with open(bundle) as f:
        newText = f.read()
        for prev, repl in bundle_rplc.items():
            newText = newText.replace(prev, repl)
    with open(bundle, "w") as f:
        f.write(newText)


def assignment_barplot(output_dir: str,
                       feature_assignments: pd.DataFrame,
                       feature_metadata: pd.DataFrame,
                       assignments_map: pd.DataFrame,
                       per_value: str) -> None:

    # subset metadata by per_value
    sub_col = assignments_map.columns[0]
    # check per value is in set of allowed
    allowed_ = set(assignments_map[sub_col].values)
    if per_value not in allowed_:
        allowed_ = ', '.join(list(allowed_))
        raise ValueError('The value given %s is not valid. Please choose from'
                         ' one of the following: %s' % (per_value, allowed_))

    # subet sample metadata
    keep_ = assignments_map[sub_col].isin([per_value])
    assignments_map = assignments_map[keep_]
    sub_ = list(assignments_map.index)
    assignments_map.index.name = 'sampleid'

    # make barplot
    _barplot(output_dir,
             feature_assignments.loc[sub_, :],
             pd.Series(feature_metadata.Taxon),
             Metadata(assignments_map))
