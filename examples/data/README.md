# Example data

The [example notebooks](../README.md) run on small samples of public datasets,
committed here so that they work offline. Each section below says where a
dataset comes from, under which license it is shared, how to credit it and
what was changed. [`datasets.json`](datasets.json) holds the same record for
the loader, with the download address and the SHA-256 checksum of every file.

[`example_data.py`](../example_data.py) reads the samples with `load(name)`.
`load(name, full=True)` downloads the whole dataset from its source, checks it
against the recorded checksum and builds the full table. The numbers in these
files were not produced by TimeGym; keep the credit lines when you pass the
files on.

## Store sales

`store-sales.csv`: units sold per week for 30 products of a UK online gift
shop, 104 weeks from 7 December 2009 to 28 November 2011. The first row holds
the shop's stock codes, the second the product names, and each later row one
week, starting on Monday. The 30 products are those with the most order lines
among the products that sold in at least 100 of the 104 weeks.

- Source: Online Retail II, UCI Machine Learning Repository,
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Changes: Order lines summed into units per product and week, with cancelled orders subtracted and totals below zero set to zero. Postage, fees and other non-product lines removed. The nine days repeated across the two sheets counted once. Weeks with no orders set to zero; the first and last weeks, which the log cuts short, dropped. Product names put in sentence case.
- Full version: every product in the order log, 4,669 of them, in the same
  weekly layout. Building it reads a spreadsheet of a million order lines,
  which takes about a minute and needs `openpyxl` (part of the `examples`
  install option).

## Bitcoin

`bitcoin.csv`: the daily Bitcoin price in US dollars, 4,021 days from 17 July
2010 to 19 July 2021.

- Source: Bitcoin Dataset without Missing Values, Monash Time Series
  Forecasting Archive, on Zenodo, <https://zenodo.org/records/5122101>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2021). Bitcoin Dataset without Missing Values (Version 1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5122101. The figures were collected from https://bitinfocharts.com.
- Changes: Converted from the archive's .tsf file to a table with one row per day. The sample keeps the price only. Days the source had no figure for were already filled with the last known value by the archive.
- Full version: the price and 17 other daily series, such as transactions,
  hash rate, search interest and tweets. A series is empty before its first
  day.

## Airline passengers

`airline-passengers.csv`: international airline passengers per month, in
thousands, 144 months from January 1949 to December 1960. The sample is the
whole series. The repository's research record keeps its own copy of the
source file in [`experiments/data`](../../experiments/data/README.md).

- Source: AirPassengers, R datasets package, as a CSV file in Rdatasets,
  <https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html>
- License: [GPL-2 | GPL-3 (R datasets package)](https://stat.ethz.ch/R-manual/R-devel/library/datasets/DESCRIPTION)
- Credit: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (1994). Time Series Analysis: Forecasting and Control (3rd ed.), series G. Distributed with R's datasets package and by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The fractional-year column replaced by the month, and the value column named passengers.

## Check or rebuild a sample

```bash
python examples/example_data.py --check                 # samples match datasets.json
python examples/example_data.py --rebuild store-sales   # download the source and write the sample again
```

Downloads are kept in `~/.cache/timegym-examples`; set `TIMEGYM_EXAMPLES_CACHE`
to use another folder.

## Add a dataset

Use data whose license allows sharing a sample with credit, and say so here.
Add its record to `datasets.json`, its reader and builder to
`example_data.py`, a sample of at most 500 KB to this folder with
`--rebuild`, and a section to this page. `tests/test_examples.py` checks that
the four agree.
