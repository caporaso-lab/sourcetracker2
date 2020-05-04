# SourceTracker2
[![Build Status](https://travis-ci.org/biota/sourcetracker2.svg?branch=master)](https://travis-ci.org/biota/sourcetracker2) [![Coverage Status](https://coveralls.io/repos/github/biota/sourcetracker2/badge.svg)](https://coveralls.io/github/biota/sourcetracker2)

SourceTracker was originally described in [Knights et al., 2011](http://www.ncbi.nlm.nih.gov/pubmed/21765408).
If you use this package, please cite the original SourceTracker paper linked
above pending publication of SourceTracker 2.

# API vs. CLI

There are two ways to access the SourceTracker 2 functionality, via the command
line (CLI) or the python API. Users seeking to replicate the functionality
of SourceTracker 1 should use the command line functionality (`sourcetracker2 gibbs`)
Programmatic users are encouraged to use the API (exposed via `gibbs`).
The help documentation is broken down into sections with separate subsections for
API and CLI usage.

# File Formats

## Command Line
For descriptions of all file formats and options, please see the help
documentation, available with the command `sourcetracker2 gibbs --help`.

This script requires a feature X sample contingency table (traditionally an OTU
table) and sample metadata (traditionally a mapping file).

The feature X sample table is an `nXm` table (`n` rows, `m` columns) with sample IDs in the first column, and feature IDs in the first row. The values in each 'cell' of the table
must be integer counts.

The sample metadata file is an `sXk` table (`s` rows, `k` columns) with sample IDs in
the first column, and metadata headers in the first row. The values in each 'cell' of the
table can be any type of data, and represent information about each sample.

Any feature table that can be read by the `biom-format >= 2.1.4` package will be
acceptable input. For example file formats please look at the test mapping file
and feature (OTU) tables we have included [here](https://github.com/biota/sourcetracker2/tree/master/data/tiny-test).

## API
For descriptions of the requirements, please see documentation in the `gibbs`
function. This function wraps the main workhorse function `gibbs_sampler`
and exposes all the parameters necessary to control the behavior of the Gibb's
sampling as well as the parallel functionality etc.

A superficial but important difference from the CLI framework is that, internally,
SourceTracker 2 represents all tables as sample X feature (samples are rows,
columns are features). This reflects choices in Dan's original code, as well as
eases metadata based subsetting of tables. The API functions expect data in
sample X column format.

# Preprocessing

## Command line
Input feature data should be counts. If non-count data (e.g. the count of
feature i in sample j was 4.63) is passed, the data will be rounded to the
nearest integer.

Rarefaction is performed by default at 1000 seqs/sample for both sinks and
sources. This is done to prevent samples with more counts from dominating the
contributions. Rarefaction depth can be set (or entirely disabled) with the ``--source_rarefaction_depth`` and ``--sink_rarefaction_depth`` parameters. Sources
samples which are collapsed are rarefied after collapse.

Samples which are not present in both the input feature table and the metadata
are excluded from the analysis.

Samples which come from the same source environment are 'collapsed', meaning
their mean value for each feature is computed and used in the analysis. See the
'Theory' section below for a discussion of this approach.

## API
The `gibbs` function does not perform collapsing or rarefaction. Validty checks
are performed on the input to the `gibbs` data. If `sources` and `sinks`
dataframes are passed, the columns of the dataframes are
checked and must overlap exactly (identical order).


# Output

## Command line
There are two default output files, the `mixing_proporitions.txt` and
`mixing_proportion_stds.txt`. `mixing_proporitions.txt` is a tab-separated contingency table with sinks as rows and sources as columns. The values in the table are the
mean fractional contributions of each source to each sink. `mixing_proporitions_stds`
has the same format, but contains the standard deviation of each fractional contribution.

Optionally, you can create per-sink feature X sample tables with the `--per_sink_feature_assignments` flag. The per-sink feature tables are labeled with
the name of the sink. For example, if we have a sink called 'hand_sample3' the
output feature table would be 'hand_sample3.feature_table.txt'. These tables record the
origin source of each sink sequence (count of a feature).

## API
The outputs of the `gibbs` function are identical to the command line outputs,
just in dataframe form.


# Documentation

This script replicates and extends the functionality of Dan Knights's
SourceTracker R package.

A major improvement in this version of SourceTracker is the ability to run it in parallel.
Currently, parallelization across a single machine is
supported for both estimation of source proportions and leave-one-out source
class prediction. The speedup from parallelization should be approximately a
factor of `jobs` that are passed. For instance, passing `--jobs 4` will
decrease runtime approximately 4X (less due to overhead). Note that you can
only specify as many jobs as you have sink samples. For instance, passing 10
jobs with only 5 sink samples will not result in the code executing any faster
than passing 5 jobs, since there is a 1 sink per job limit. Said another way,
a single sink sample cannot be split up into multiple jobs.

# Installation

SourceTracker2 is Python 3 software. The easiest way to install it is using Anaconda. If you don't already have Anaconda installed, you can install it following [their install instructions](https://docs.continuum.io/anaconda/install).

To install SourceTracker 2 using Anaconda, run the following commands:

```bash
conda create -n st2 -c biocore python=3.5 numpy scipy scikit-bio biom-format h5py hdf5 seaborn
conda activate st2
pip install sourcetracker
```

If you wish to use the latest features in the development version of sourcetracker2
you can install it directly from the github repository.

**Note: Only run this if you wish to use the development version**
```bash
pip install https://github.com/biota/sourcetracker2/archive/master.zip
```

To test that your installation was successful, try the following command:

```bash
sourcetracker2 gibbs --help
```

When you open a new terminal, you'll always need to run:

```bash
source activate st2
```

to activate your SourceTracker 2 environment.

# Theory

This document describes some of the basic theory for use of SourceTracker2. For
more theory and a visual walkthrough, please see the [Juypter notebook](https://github.com/biota/SourceTracker_rc/blob/master/ipynb/Sourcetracking%20using%20a%20Gibbs%20Sampler.ipynb).

There are main two ways to use this script:  
 (1) Estimating the proportions of different (microbial) sources to a sample of
     a (microbial) sink.  
 (2) Using a leave-one-out (LOO) strategy, predict the metadata class of a
     given (microbial) sample.  

The main functionality (1) is, the estimation of the proportion of `sources`
to a given `sink`. A `source` and a `sink` are both vectors of feature
abundances. A  `source` is typically multiple samples that come from
an environment of interest, a `sink` is usually a single sample.

As an example, consider the classic balls in urns problem. There are three urns, each
of which contains a set of differently colored balls in different proportions.
Imagine that you reach into Urn1 and remove `u_1` balls without looking at the
colors. You reach into Urn2 and remove `u_2` balls, again without looking at
the colors. Finally, you reach into Urn3 and remove `u_3` balls, without
looking at the colors. You then mix your individual samples (`u_1`, `u_2`,
and `u_3`) and produce one mixed sample whose color counts you survey.

|        | Urn1 | Urn2 | Urn3 | Sample |
|--------|:----:|------|------|--------|
| Blue   |   3  | 6    | 100  | 26     |
| Red    |  55  | 12   | 30   | 9      |
| Green  |  10  | 0    | 1    | 1      |
| ...    |      |      |      |        |
| Orange | 79   | 18   | 0    | 50     |


Your goal is to recover the numbers `u_1`, `u_2`, and `u_3` using only the
knowledge of the colors of your mixed sample and the proportions of each color
in the sinks.

The Gibb's sampler is a method for estimating `u_1`, `u_2` and `u_3`. In
SourceTracker2, the Gibb's sampler would be used to make an
estimate of the source proportions (`u_1`, `u_2`, and `u_3`) plus an
estimate of the proportion of balls in the sample that came from an unknown
source. In the urns example there is no unknown source; all of the balls came from
one of the three urns. In a real set of microbial samples, however, it is common that the
source samples assayed are not the only source of microbes found in the sink
sample (air, skin, soil or other microbial sources that were not included).

In practice, researchers often take multiple samples from a given source
environment (e.g. to learn the true distribution of features in the source). It
is desirable to 'collapse' samples from the same source into one representative
source sample. This is mainly for interpretability. Consider the urn example
above. In practice we would not know the exact contents of any of the urns.
The only way to learn the actual source distributions would be to sample them
repeatedly. Combining n samples from urn 1 into a single source would make the
estimate of the urns true proportions of different colors more accurate and
would make interpreting the results easier; there would be only 3 source
proportions, plus the unknown, to interpret rather than 3n+1 (assuming n samples from each
urn). Please read about (2) below to understand an important
limitation of this collapsing processes.

A second function of of this script is (2), the prediction of the metadata class
of sample based on the feature abundances in all samples and the metadata
classes of all samples.

In practice, this function is useful for checking whether the source groupings
you have computed are good groupings. As an example, imagine that you are baking
bread and want to know where the microbes are coming from in your dough.
You think there are three main sources: flour, water, hands. You take 10 samples
from each of those environments (10 different bags of flour, 10 samples from
water, 10 samples from your hands on different days). For computing source
proportions, you would normally collapse each of the 10 samples from the given
class into one source (so you'd end up with a 'Flour', 'Water', and 'Hand'
source). However, if the flour you use comes from different facilities, it is
likely that the samples will have very different microbial compositions. If this is the
case, collapsing the flour samples into a single source would be inappropriate,
since there are at least two different sources of microbes from the
flour. To check the homogeneity of your source classifications, you can use the
LOO strategy to make sure that all sources within each class look the same.

# Usage examples

These usage examples expect that you are in the directory  
`sourcetracker2/data/tiny-test/`

**Calculate the proportion of each source in each sink**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o example1/`

**Calculate the proportion of each source in each sink using an alternate sample metadata mapping file where samples are described differently.**  
`sourcetracker2 gibbs -i otu_table.biom -m alt-map.txt -o example2/ --source_sink_column source-or-sink --source_column_value src --sink_column_value snk --source_category_column sample-type`

**Calculate the class label (i.e. 'Env') of each source using a leave one out
strategy**    
`sourcetracker2 gibbs -i otu_table.biom -m map.txt --loo -o example3/`

**Calculate the proportion of each source in each sink, using 100 burnins**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o example4/ --burnin 100`

**Calculate the proportion of each source in each sink, using a sink
rarefaction depth of 1700 and source rarefaction depth of 1500**    
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o example5/ --sink_rarefaction_depth 1700 --source_rarefaction_depth 1500`

**Calculate the proportion of each source in each sink, parallel with 5 jobs**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o example6/ --jobs 5`

**Calculate the proportion of each source in each sink, with 5 jobs. Write the per sink feature tables (what SourceTracker 1 called
'full output'). These are feature by sample table indicating the origin source of each sequence (each count of a feature).**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o example7/ --jobs 5 --per_sink_feature_assignments`


# Miscellaneous

The current implementation of SourceTracker 2 does not contain functionality for
auto-tuning of the parameters (`alpha1`, `alpha1`, etc.). 

For auto-tuning functionality, please see the original R code.
