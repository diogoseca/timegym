# Examples

Forecasting jobs on real public data, from the first model to a decision, with
TimeGym in the loop. Read them with their results on the
[examples page](https://diogoseca.github.io/timegym/docs/examples/) of the
documentation, or run them here.

| Example | The job | What TimeGym showed |
| --- | --- | --- |
| [Weekly shop sales: from forecast to stock order](store-sales.ipynb) | Restock 30 products of an online gift shop every four weeks | Why the usual regression loses to a plain average, which simpler forecaster fits, and that neither knows about Christmas |
| [Spare parts: the best score goes to a forecast of zero](spare-parts.ipynb) | Decide how many of each slow-selling car part to hold for six months | The lowest average error belongs to a forecast of zero; a score in the parts counter's terms and the scorecard name the forecaster that keeps parts on the shelf |
| [The Christmas order: five Decembers cannot pick the forecaster](christmas-order.ipynb) | Order a souvenir shop's Christmas stock three months ahead, with seven years of history | A straight line fitted in units forecasts the last two Decembers at half; `target_transform=LogTransform()` puts it level with the best; generated shops with a known kind of growth show what a five-December backtest can tell, what each choice costs and how much to order on top |
| [Call center staffing: forecast again at ten o'clock](call-center.ipynb) | Plan a bank's call center agents half hour by half hour | What a newer forecast is worth: nothing from two weeks ahead to the evening before, 15% once it looks at how yesterday went, another 12% from the calls taken by 10:00; and counting rows to find "last week" costs 18% on a calendar with holidays |
| [Train station riders: what is the weather worth, at best?](station-riders.ipynb) | Decide whether a ridership forecast for 20 train stations needs a feed of weather forecasts | Told the weather that really came, the forecaster gains under 2% on weekdays and 8% on weekends, from 24% at one station to nothing at two; "tomorrow like today" keeps half; the same cases and scoring with one table added |
| [Tomorrow's electricity demand: let the worst days say what is missing](electricity-demand.ipynb) | Forecast tomorrow's demand, half-hour by half-hour | Which days a day-ahead forecast gets wrong, round by round: the weekday, then public holidays, then weather the data lacks |
| [Solar farm output: a range for every half hour is not a range for the day](solar-farm.ipynb) | Give a range for a solar farm's energy over the whole of tomorrow | Forecasts made of samples go through the pipeline; the half-hour score cannot tell whole days from shuffled half hours and the two whole-day scores can; generated series with the true answer show when the gap opens and what 28 samples cost |
| [River flow: checking the range, not only the forecast](river-flow.ipynb) | Give a three-day range for a river's flow | A 90% range that holds on average but not month by month, the one argument that repairs it, and how far it fails when the river rises |
| [Will it rain tomorrow? Score the forecast you already have](rain.ipynb) | Give the chance of rain tomorrow in Sydney | Our best forecaster gains 11% on "it rains one day in three" and the weather service's published forecast 53%; why the forecaster that is right most often scores worst, and which days have no score |
| [Tomorrow's temperature: which city should Sydney watch?](daily-temperature.ipynb) | Forecast Sydney's high for the next three days | Another city's record helps: Melbourne cuts tomorrow's miss by 14%, Perth helps most for later days, Darwin makes it worse, and two cities give what eight give |
| [Tomorrow's air in London: the gaps someone filled in for you](air-quality.ipynb) | Forecast the next 24 hours of fine particles at 24 monitors | A table with its gaps filled in makes every forecast look a fifth better than it is and ranks the silent monitors as the easiest; cutting cases only where the answers were measured gives the real numbers |
| [The sunspot cycle: how far ahead is a forecast worth making?](sunspots.ipynb) | Forecast a cycle eleven years ahead | A regression that leads for two years and adds nothing after, and how little room an irregular cycle leaves a cleverer model |
| [El Niño: how far ahead depends on the month you ask](el-nino.ipynb) | Forecast the Southern Oscillation Index nine months ahead and say which of those months to act on | "Seven months" on average hides two cases: a forecast that has to pass from April into May gains nothing, even one month ahead, and one that does not is still ahead at nine; checked on earlier years and on generated series with no seasons |
| [The forest that cannot follow growth](passenger-growth.ipynb) | Forecast next year's airline passengers | A random forest that passes its backtest cannot forecast a trend, and the textbook fix makes it worse elsewhere |
| [You tried 50 models: is the winner real?](best-of-fifty.ipynb) | Choose a model for next year's patient numbers of 114 medical products | The backtest's winner leads by 0.2%, nine others cannot be told apart from it, a winning score flatters by about 4%, and the next year the winner comes tenth |
| [A traffic jump: is this the new normal?](traffic-jump.ipynb) | Three days into a jump in a page's views, forecast the next four weeks | The same forecasters on every week and on the alarm days give opposite orders; two scorecard rows name what a jump can do; a forecaster that learns the fade from earlier jumps beats all four |
| [Site capacity: which forecast still works four weeks ahead?](site-capacity.ipynb) | Book a site's capacity four weeks ahead, with spare capacity on top | A generated weekly cycle catches a one-day slip the real data hides; the same weeks cut at five lead times show two forecasters level for next week and far apart for the job; the kept forecasts size the spare capacity |
| [Bitcoin: finding out you have nothing](bitcoin.ipynb) | Forecast tomorrow's price | A 79% hit rate that came from one input reaching into the future; the honest model ties with yesterday's price |
| [The S&P 500 next week: the direction and the size are different questions](sp500.ipynb) | Say something useful each Friday about the index's next week | No forecaster calls the direction better than "always up"; a 90% range that follows the last 26 weeks holds in crashes, where one built from all history held on 60% of the weeks of 2008 |

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
