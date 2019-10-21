import os
import shutil
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
        mf_samples = sample_metadata[sample_metadata[category_column].isin(proportions.columns)]
        mf_samples = mf_samples.set_index(category_column)
        mf_samples = mf_samples.loc[~mf_samples.index.duplicated(keep='first')]
        mf_samples[category_column] = list(mf_samples.index)
        mf_samples = mf_samples[mf_samples.columns[::-1]]
        mf_samples.index.name = 'sampleid'

    # make the feature metadata (mock taxonomy)
    mf_feature = sample_metadata[sample_metadata[category_column].isin(proportions.index)]
    mf_feature = mf_feature.set_index(category_column)
    mf_feature = mf_feature.loc[~mf_feature.index.duplicated(keep='first')]
    mf_feature.loc['Unknown', :] = 'Unknown'
    mf_feature[category_column] = list(mf_feature.index)
    mf_feature = mf_feature[mf_feature.columns[::-1]]
    mf_feature = mf_feature.astype(str).apply(lambda x: '; '.join(x), axis=1)
    mf_feature = pd.DataFrame(mf_feature,
                            columns = ['Taxon'])
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
    bundle_rplc = {'Relative Frequency':'Source Contribution',
                'Taxonomic Level':'Source Grouping',
                'Sample':'Sink'}
    # make small text chnage to bundle
    with open(bundle) as f:
        newText=f.read()
        for prev, repl in bundle_rplc.items():
            newText = newText.replace(prev, repl)
    with open(bundle, "w") as f:
        f.write(newText)


def assignment_barplot(output_dir: str,
                       feature_assignments: pd.DataFrame,
                       feature_metadata: pd.DataFrame,
                       sample_metadata: Metadata,
                       per_value: str,
                       category_column: str = DEFAULT_CAT) -> None:

    # scriptable metadata
    sample_metadata = sample_metadata.to_dataframe()

    # un-merge by the sink
    feature_assignments['sink'] = [sink.split('-')[0]
                                    for sink in feature_assignments.index]
    fas_unmerged = {sink:source_df.drop(['sink'], axis=1)
                    for sink, source_df in feature_assignments.groupby('sink')}

    if per_value not in fas_unmerged.keys():
        allowed_ = ', '.join(fas_unmerged.keys())
        raise ValueError('The value given %s is not valid. Please choose from'
                         ' one of the following: %s'%(per_value, allowed_))
    # grab sink and source
    source_df  = fas_unmerged[per_value]
    # subset the sample metadata
    keep_ = [cat.split('-')[1] for cat in source_df.index]
    source_df.index = keep_
    mf_sub = sample_metadata[sample_metadata[category_column].isin(keep_)]
    mf_sub = mf_sub.set_index(category_column)
    mf_sub = mf_sub.loc[~mf_sub.index.duplicated(keep='first')]
    mf_sub.loc['Unknown', :] = 'Unknown'
    mf_sub.index.name = 'sampleid'

    # make barplot
    _barplot(output_dir,
            source_df,
            pd.Series(feature_metadata.Taxon),
            Metadata(mf_sub))
