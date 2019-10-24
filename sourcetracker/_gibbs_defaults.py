# Configuration file where you can set the parameter default values and
# descriptions.
DEFAULT_ALPH1 = .001
DEFAULT_ALPH2 = .1
DEFAULT_ONE = 1
DEFAULT_TEN = 10
DEFAULT_HUND = 100
DEFAULT_THOUS = 1000
DEFAULT_FLS = False
DEFAULT_TRU = True
DEFAULT_SNK = 'SourceSink'
DEFAULT_SRS = 'source'
DEFAULT_SRS2 = 'sink'
DEFAULT_CAT = 'Env'

DESC_TBL = 'Path to input table.'
DESC_MAP = 'Path to sample metadata mapping file.'
DESC_OUT = 'Path to the output directory to be created.'
DESC_LOO = ('Classify each sample in `sources` using a leave-one-out '
            'strategy. Replicates -s option in Knights et al. '
            'sourcetracker.')
DESC_JBS = 'Number of processes to launch.'
DESC_ALPH1 = ('Prior counts of each feature in the training '
              'environments. Higher values decrease the trust in the '
              'training environments, and make the source environment '
              'distributions over taxa smoother. A value of 0.001 '
              'indicates reasonably high trust in all source '
              'environments, even those with few training sequences. A '
              'more conservative value would be 0.01.')
DESC_ALPH2 = ('Prior counts of each feature in the `unknown` environment'
              ' as a fraction of the counts of the current sink being '
              'evaluated. Higher values make the `unknown` environment '
              'smoother and less prone to overfitting given a training '
              'sample.')
DESC_BTA = ('Count to be added to each feature in each environment, '
            'including `unknown` for `p_v` calculations.')
DESC_RAF1 = ('Depth at which to rarify sources. If 0, no '
             'rarefaction performed.')
DESC_RAF2 = ('Depth at which to rarify sinks. If 0, no '
             'rarefaction performed.')
DESC_RST = ('Number of independent Markov chains to grow. '
            '`draws_per_restart` * `restarts` gives the number of '
            'samplings of the mixing proportions that will be '
            'generated.')
DESC_DRW = ('Number of times to sample the state of the Markov chain '
            'for each independent chain grown.')
DESC_BRN = ('Number of passes (withdarawal and reassignment of every '
            'sequence in the sink) that will be made before a sample '
            '(draw) will be taken. Higher values allow more '
            'convergence towards the true distribtion before draws '
            'are taken.')
DESC_DLY = ('Number passes between each sampling (draw) of the '
            'Markov chain. Once the burnin passes have been made, a '
            'sample will be taken, and then taken again every `delay` '
            'number of passes. This is also known as `thinning`. '
            'Thinning helps reduce the impact of correlation between '
            'adjacent states of the Markov chain.')
DESC_PFA = ('If True, this option will cause SourceTracker2 to write '
            'out a feature table for each sink (or source if `--loo` '
            'is passed). These feature tables contain the specific '
            'sequences that contributed to a sink from a given '
            'source. This option can be memory intensive if there are '
            'a large number of features. Note: in the QIIME 2 plugin'
            ' this is non-optional and always set to true.')
DESC_RPL = ('Sample with replacement instead of '
            'sample without replacement')
DESC_SNK = ('Sample metadata column indicating which samples should be'
            ' treated as sources and which as sinks.')
DESC_SRS = ('Value in source_sink_column indicating which samples '
            'should be treated as sources.')
DESC_SRS2 = ('Value in source_sink_column indicating which samples '
             'should be treated as sinks.')
DESC_CAT = ('Sample metadata column indicating the type of each '
            'source sample.')
OUT_MEAN = ('The mixing_proporitions output is a table with sinks'
            ' as rows and sources as columns. The values in the '
            'table are the mean fractional contributions of each '
            'source to each sink.')
OUT_STD = ('The mixing_proporitions_*stds* has the same format as '
           'mixing proporitions, but contains the standard deviation'
           ' of each fractional contribution.')
OUT_PFA = ('The feature table for each sink (or source if `--loo '
           'is passed). This feature table contains the specific '
           ' of each fractional contribution.')
DESC_FMAP = ('Taxonomic annotations for features in the provided '
             'feature table. All features in the feature table must'
             'have a corresponding taxonomic annotation. Taxonomic '
             'annotations that are not present in the feature table '
             'will be ignored. ')
DESC_PVAL = ('The value of the sink (or source if `--loo` '
             'is passed) for the desired barplot visualization.')
OUT_PFAM = ('The mapping file to the per feature table for each '
            'sink (or source if `--loo is passed). '
            'This feature table contains the specific '
            ' of each fractional contribution.')
