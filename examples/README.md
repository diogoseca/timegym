# Examples

Forecasting jobs on real public data, from the first model to a decision, with
TimeGym in the loop. Read them with their results on the
[examples page](https://diogoseca.github.io/timegym/docs/examples/) of the
documentation, or run them here.

| Example | The job | What TimeGym showed |
| --- | --- | --- |
| [Weekly shop sales: from forecast to stock order](store-sales.ipynb) | Restock 30 products of an online gift shop every four weeks | Why the usual regression loses to a plain average, which simpler forecaster fits, and that neither knows about Christmas |
| [Spare parts: the best score goes to a forecast of zero](spare-parts.ipynb) | Decide how many of each slow-selling car part to hold for six months | The lowest average error belongs to a forecast of zero; a score in the parts counter's terms and the scorecard name the forecaster that keeps parts on the shelf |
| [Tomorrow's electricity demand: let the worst days say what is missing](electricity-demand.ipynb) | Forecast tomorrow's demand, half-hour by half-hour | Which days a day-ahead forecast gets wrong, round by round: the weekday, then public holidays, then weather the data lacks |
| [River flow: checking the range, not only the forecast](river-flow.ipynb) | Give a three-day range for a river's flow | A 90% range that holds on average but not month by month, the one argument that repairs it, and how far it fails when the river rises |
| [Will it rain tomorrow? Score the forecast you already have](rain.ipynb) | Give the chance of rain tomorrow in Sydney | Our best forecaster gains 11% on "it rains one day in three" and the weather service's published forecast 53%; why the forecaster that is right most often scores worst, and which days have no score |
| [Tomorrow's temperature: which city should Sydney watch?](daily-temperature.ipynb) | Forecast Sydney's high for the next three days | Another city's record helps: Melbourne cuts tomorrow's miss by 14%, Perth helps most for later days, Darwin makes it worse, and two cities give what eight give |
| [The sunspot cycle: how far ahead is a forecast worth making?](sunspots.ipynb) | Forecast a cycle eleven years ahead | A regression that leads for two years and adds nothing after, and how little room an irregular cycle leaves a cleverer model |
| [The forest that cannot follow growth](passenger-growth.ipynb) | Forecast next year's airline passengers | A random forest that passes its backtest cannot forecast a trend, and the textbook fix makes it worse elsewhere |
| [You tried 50 models: is the winner real?](best-of-fifty.ipynb) | Choose a model for next year's patient numbers of 114 medical products | The backtest's winner leads by 0.2%, nine others cannot be told apart from it, a winning score flatters by about 4%, and the next year the winner comes tenth |
| [Bitcoin: finding out you have nothing](bitcoin.ipynb) | Forecast tomorrow's price | A 79% hit rate that came from one input reaching into the future; the honest model ties with yesterday's price |

## Run them

From a clone of the repository:

```bash
python -m pip install -e '.[examples]'
python -m jupyterlab examples
```

Every notebook reads a small sample committed in [`data/`](data/README.md), so
it runs offline, and is saved with its outputs. Installing the wheel alone does
not install these notebooks.

## The data

[`example_data.py`](example_data.py) loads the datasets:

```python
from example_data import load, about

sales = load("store-sales")                   # the committed sample
everything = load("store-sales", full=True)   # downloads the public source once
print(about("store-sales"))                   # source, license and credit
```

[`data/README.md`](data/README.md) gives each dataset's source, license, credit
and checksums. `python examples/example_data.py --check` verifies the committed
samples, and `--rebuild NAME` writes one again from its source.

## Add an example

[Contributing](../docs/contributing.md#add-an-example) lists the steps.

## Terminal scripts

Two scripts from before the notebooks, still used by the tests:

- [`quickstart.py`](quickstart.py): generate controlled cases and inspect a
  diagnostic report.
- [`reproduce_report.py`](reproduce_report.py): replay a saved report case for a
  supported built-in model.

```bash
python examples/quickstart.py
python examples/reproduce_report.py report.json --case-index 0
```

The README's figures are redrawn by
[`tools/readme_figures.py`](../tools/readme_figures.py). The original example
notebook is preserved as
[Quick Start (2021)](https://github.com/diogoseca/timegym/blob/v1.4.0/source/archive/Quick%20Start%20%282021%29.ipynb) at the v1.4.0 tag.
