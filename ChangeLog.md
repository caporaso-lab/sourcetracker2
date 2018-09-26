# sourcetracker2 changelog

## 2.0.1-dev (changes since 2.0.1 go here)

 * Updated subsample function to be compatible with new version of pandas
 * A unified API for sourcetracking with Gibbs sampling, including
   leave-one-out cross-validation, has been created and is accessible as
   ``sourcetracker.gibbs``.
 * Heatmap plotting of mixing proportion means has been added to the output of
   the command line interface, and to the public API as ``sourcetracker.gibbs.plot_heatmap``.
 * The per-sink feature assignments are recorded for every run and written to
   the output directory. They are named as ``X.contingency.txt`` where ``X``
	is the name of a sink.

 * Sample with replacement functionality has been added

## 2.0.1

  * Initial alpha release.
  * Re-implements the Gibbs sampler from [@danknights's SourceTracker.](https://github.com/danknights/sourcetracker).
  * [click](http://click.pocoo.org/)-based command line interface through the ``sourcetracker2`` command.
  * Supports parallel execution using the `--jobs` parameter.
