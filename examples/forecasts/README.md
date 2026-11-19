# Saved forecasts

One example, [Ready-made model or your own?](../ready-made-model.ipynb),
compares our forecasters with a large model that someone else trained.
Running that model takes PyTorch and a download of 478 MB, so the example
reads forecasts the model made in advance. This folder holds them.

| File | What it holds |
| --- | --- |
| `ready-made-model.csv` | 904 forecasts of 12 months each, one row per history. `history` is a fingerprint of the history the forecast was made from: the first 12 characters of the SHA-1 of its values, rounded to 4 decimals. Columns `1` to `12` are the forecast for the 12 months that follow, to six significant digits |
| `ready-made-scorecard.json` | The model's scorecard on 600 generated series (ten patterns at two noise levels, 30 series each, five years of history and one year ahead), as written by `Scorecard.save_baseline` |

## The model

- [Chronos-2](https://huggingface.co/amazon/chronos-2) (`amazon/chronos-2`),
  revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, published by Amazon
  under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Run on 19 November 2026 with `chronos-forecasting` 2.3.2 and PyTorch 2.15.0
  on a CPU, under Linux.
- Each forecast is the middle of the range the model returns (its 0.5
  quantile). The model was used as published: nothing was fitted to the data.

## The histories

- 684 forecasts are for the 114 products of the
  [hospital patients sample](../data/README.md#hospital-patients) (CC BY 4.0):
  years 5, 6 and 7, each from all the months before it and from the last 24
  months only.
- 220 are for series the notebook generates: a level that rises for a time and
  comes back.

## Make them again

```bash
python -m pip install -e '.[examples]' chronos-forecasting==2.3.2
```

Set `RUN_MODEL = True` in the notebook's first code cell and run all cells.
The notebook then asks the model for every forecast, which takes a minute or
two, and its last code cell writes both files. The first run downloads the
model from Hugging Face.

Another machine or another version of PyTorch can change the last digits of a
forecast. The scorecard file was written on Linux; on another system a few of
the generated series can be drawn differently, so a comparison with it can
move by a cell or two.
