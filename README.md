[![Build Status](https://travis-ci.org/biota/sourcetracker2.svg?branch=master)](https://travis-ci.org/biota/sourcetracker2)

# SourceTracker2

SourceTracker was originally described in [Knights et al., 2011](http://www.ncbi.nlm.nih.gov/pubmed/21765408).
If you use this package, please cite the original SourceTracker paper linked
above.

# Documentation

This script replicates and extends the functionality of Dan Knights's
SourceTracker R package.

The `mapping file` which describes the `sources` and `sinks` must be
formatted in the same way it was for the SourceTracker R package. Specifically,
there must be a column `SourceSink` and a column `Env`. For an example, look
at `sourcetracker2/data/tiny-test/`.

A major improvement in this version of SourceTracker is the ability to run it in parallel.
Currently, parallelization across a single machine is
supported for both estimation of source proportions and leave-one-out source
class prediction. The speedup from parallelization should be approximately a
factor of `jobs` that are passed. For instance, passing `--jobs 4` will
decrease runtime approximately 4X (less due to overhead). The package
`ipyparallel` is used to enable parallelization. Note that you can only specify
as many jobs as you have sink samples. For instance, passing 10 jobs with only 5
sink samples will not result in the code executing any faster than passing 5 jobs,
since there is a 1 sink per job limit. Said another way, a single sink sample
cannot be split up into multiple jobs.

# Installation

SourceTracker2 is Python 3 software. The easiest way to install it is using Anaconda. If you don't already have Anaconda installed, you can install it following [their install instructions](https://docs.continuum.io/anaconda/install).

To install SourceTracker 2 using Anaconda, run the following commands:

```bash
conda create -n st2 python=3.5 numpy scipy h5py hdf5
source activate st2
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

This readme describes some of the basic theory for use of SourceTracker2. For
more theory and a visual walkthrough, please see the [juypter notebook](https://github.com/biota/SourceTracker_rc/blob/master/ipynb/Sourcetracking%20using%20a%20Gibbs%20Sampler.ipynb).

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
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o mixing_proportions/`

**Calculate the class label (i.e. 'Env') of each source using a leave one out
strategy**    
`sourcetracker2 gibbs -i otu_table.biom -m map.txt --loo -o source_loo/`

**Calculate the proportion of each source in each sink, using 100 burnins**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o mixing_proportions/ --burnin 100`

**Calculate the proportion of each source in each sink, using a sink
rarefaction depth of 2500**    
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o mixing_proportions/ --sink_rarefaction_depth 2500`

**Calculate the proportion of each source in each sink, using ipyparallel to run in parallel with 5 jobs**  
`sourcetracker2 gibbs -i otu_table.biom -m map.txt -o mixing_proportions/ --jobs 5`

# Miscellaneous

The current implementation of SourceTracker 2 does not contain functionality for
visualization of results or autotuning of the parameters (`alpha1`, `alpha1`,
etc.). For an example of how you might visualize the data, please see
this [juypter notebook](https://github.com/biota/SourceTracker2/blob/master/ipynb/Visualizing%20results.ipynb).
For autotuning functionality, please see the original R code.

Like the old SourceTracker, SourceTracker2 rarifies the source environments it
collapses by default.
