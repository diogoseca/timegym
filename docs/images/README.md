# Documentation images

The figures shown in the project README and on the documentation site. Each
section says what draws it; redraw a figure there, never by hand.

## Alignment preview

`alignment-demo.png` is saved by the index-shift section of
[`03_faults_repairs_and_replay.ipynb`](../../tutorials/03_faults_repairs_and_replay.ipynb)
(cell `05-alignment-plot`): a seasonal forecast with its dates shifted by one
step, against the dates asked for. It is shown in the project README. Run that notebook to
refresh it.

## Scorecard preview

`scorecard.png` is the scorecard of `LastValueForecaster`, drawn by
[`tools/readme_figures.py`](../../tools/readme_figures.py):

```bash
python tools/readme_figures.py
```

[13 · Scorecard](../../source/13_scorecard.ipynb) explains how it is built.

## README figures

`readme-trend-forecasts.png` and `readme-trend-errors.png` show the output of
the README's "More things to do with a scorecard". Redraw both after a
plotting change with
[`tools/readme_figures.py`](../../tools/readme_figures.py):

```bash
python tools/readme_figures.py
```
