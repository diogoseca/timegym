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

## Spare parts

`spare-parts.csv`: units sold per month for 314 car parts, 51 months from
January 1998 to March 2002. There is one column per part, under the archive's
labels (`T8`, `T20` and so on); the archive does not say what each part is.
The 314 are every eighth part among the 2,509 that have a figure for every
month.

- Source: Car Parts Dataset (with Missing Values), Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/4656022>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). Car Parts Dataset (with Missing Values) (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656022. The archive took the series from the R package expsmooth.
- Changes: Converted from the archive's .tsf file to a table with one row per month and one column per part, under the archive's part labels. The sample keeps every eighth part among the 2,509 that have a figure for every month.
- Full version: all 2,674 parts. 165 of them have no figures from early 1999 on,
  and those months are empty.

## Bank calls

`bank-calls.csv`: calls handled by the call center of a large North American
bank in each half hour from 7:00 to 21:00, on 164 working days from 3 March
to 24 October 2003: 4,592 half hours. The table has rows only for the times
the center was open, so there are none for nights and weekends, and six
weekdays in that span have no rows either (4 and 7 April, 26 May, 4 July,
1 September and 14 October). The index therefore has no regular step, and
`datasets.json` records that with `"regular": false`.

- Source: bank_calls, R package fpp3, as a CSV file in Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/bank_calls.html>
- License: [GPL-3 (R package fpp3)](https://github.com/robjhyndman/fpp3-package/blob/master/DESCRIPTION), the license
  of the package the table ships in. The package names Jonathan Weinberg as
  the source of the series and does not give the bank's name.
- Credit: Hyndman, R. J., and others. fpp3: Data for "Forecasting: Principles and Practice" (3rd Edition) (R package), dataset bank_calls. The package gives Jonathan Weinberg as the source of the series, which is studied in Weinberg, J., Brown, L. D., & Stroud, J. R. (2007). Bayesian forecasting of an inhomogeneous Poisson process with applications to call center data. Journal of the American Statistical Association, 102(480), 1185-1198. Distributed by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The row-number column dropped; times kept as the center's clock time, without the time zone mark the file carries; the columns named time and calls. The sample adds the five-minute counts up to half hours and leaves out the last five minutes of each day, 21:00 to 21:05.
- Full version: the calls of every five minutes from 7:00 to 21:05 on the
  same 164 days, 27,716 rows.

## City temperatures

`city-temperatures.csv`: the highest temperature of each day, in degrees
Celsius, in eight Australian cities, 725 days from 2 May 2015 to
25 April 2017. The columns are Sydney (Bureau of Meteorology station 66062),
Melbourne (86282, the airport), Adelaide (23034, the airport), Canberra
(70351), Hobart (94029), Perth (9021, the airport), Brisbane (40842, the
airport) and Darwin (14015, the airport). It comes from the same source file
as the [rain](#rain) table, where the 24 hours of a row are explained.

- Source: Temperature Rain Dataset with Missing Values, Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/5129073>
- License: [CC BY-SA 3.0 AU](https://creativecommons.org/licenses/by-sa/3.0/au/), for the reason
  given under [Rain](#rain).
- Credit: Bureau of Meteorology, Australian Government: Historical rainfall and temperature forecast and observations hourly data - weather forecasting verification data (2015-05 to 2016-04), and Rainfall and temperature forecast and observations - verification 2016-05 to 2017-04, both on data.gov.au. Put together as one file by Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2021). Temperature Rain Dataset with Missing Values (Version 1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5129073
- Changes: The highest temperature of each day taken from the archive's .tsf file; its other series left out. The sample keeps eight stations and names each column after its city.
- Full version: all 422 stations, under the Bureau's station numbers. Some
  stations have days without a figure, and those cells are empty.

## Electricity demand

`electricity-demand.csv`: electricity demand in Victoria, Australia, for every
half-hour of 2014, in gigawatts. The file has one row per day and one column
per half-hour, and `load` returns one row per half-hour. The sample is the
whole series. The source does not say how the clock change for daylight saving
time was handled.

- Source: Electricity Demand (Elecdemand) Dataset, Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/4656069>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). Electricity Demand (Elecdemand) Dataset (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656069. The archive took the series from the R package fpp2.
- Changes: Converted from the archive's .tsf file and rounded to three decimals, which is one megawatt. The file holds one row per day and one column per half-hour; the loader returns one row per half-hour.

## Hospital patients

`hospital-patients.csv`: the number of patients per month for 114 medical
products, 84 months from January 2000 to December 2006. There is one column
per product, under the archive's labels (`T3`, `T21` and so on); the archive
does not say what each product is. The 114 are every second product among the
227 with at least 100 patients a month on average.

- Source: Hospital Dataset, Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/4656014>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). Hospital Dataset (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656014. The archive took the series from the R package expsmooth.
- Changes: Converted from the archive's .tsf file to a table with one row per month and one column per product, under the archive's labels. The sample keeps every second product among the 227 with at least 100 patients a month on average.
- Full version: all 767 products. Most of them have far fewer patients a
  month; the smallest average 10.

## London air

`london-air.csv`: fine particles in the air (PM2.5, in micrograms per cubic
meter) for every hour at 24 air quality monitors in London, 3,624 hours from
1 November 2017 to 31 March 2018. Each column is one monitor, under the
network's site code. A cell is empty where the monitor reported nothing for
that hour, which is 23% of the cells: three monitors (`LH0`, `HR1` and `BX1`)
report nothing in these five months. The values are the archive's, with
nothing filled in or removed, and 0.7% of them are below zero.

- Source: KDD Cup Dataset (with Missing Values), Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/4656719>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the archive's
  license for its file. The archive does not say under which terms the
  London Air Quality Network shared the readings with the competition.
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). KDD Cup Dataset (with Missing Values) (Version 4) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656719. The archive took the series from the data of the KDD Cup 2018 forecasting competition; the London readings in it were made by the London Air Quality Network.
- Changes: The fine-particle (PM2.5) series of the 24 London monitors taken from the archive's .tsf file, one column per monitor under the network's site code; the Beijing series and the other measurements left out. Nothing filled in. The sample keeps the last five months.
- Full version: the same 24 monitors for all 15 months from 1 January 2017,
  10,920 hours. One monitor, `CT2`, starts on 1 March 2017.

The archive also publishes this dataset
[without missing values](https://zenodo.org/records/4656756): a monitor's
empty hours before its first reading are set to zero and every later empty
hour repeats the last reading. Applying that rule to the file used here gives
the archive's filled file value for value, for all 270 series (checked on
17 November 2026). The [air quality example](../air-quality.ipynb) shows what
that does to a forecast's score.

## Rain

`rain.csv`: daily weather at Sydney (Observatory Hill), Bureau of Meteorology
station 66062, 725 days from 2 May 2015 to 25 April 2017. Each row has the
rain that fell, in millimeters (`rain_mm`), the highest and lowest
temperature, in degrees Celsius, and the chance of rain, in percent, that the
Bureau gave for that day in the forecast it issued on the day itself and in
those it issued one to five days before (`bureau_same_day` to
`bureau_5_days_before`). The chance is for more than 0.2 millimeters of rain.

A row is the 24 hours that start at 15:00 on its date. The source does not
name the time zone; read as universal time, that is 1 am the next morning in
Sydney. Each forecast column has 10 days with no figure in the source, and
the first days of the table are too early to have a forecast from days
before; those cells are empty.

- Source: Temperature Rain Dataset with Missing Values, Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/5129073>
- License: [CC BY-SA 3.0 AU](https://creativecommons.org/licenses/by-sa/3.0/au/). This is the
  license data.gov.au lists for the Bureau's data, which asks for credit and
  for copies and changed versions to be shared under the same license. The
  archive lists its own file as CC BY 4.0. We follow the stricter of the two.
- Credit: Bureau of Meteorology, Australian Government: Historical rainfall and temperature forecast and observations hourly data - weather forecasting verification data (2015-05 to 2016-04), and Rainfall and temperature forecast and observations - verification 2016-05 to 2017-04, both on data.gov.au. Put together as one file by Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2021). Temperature Rain Dataset with Missing Values (Version 1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5129073
- Changes: Rain, highest and lowest temperature and the chance of rain taken from the archive's .tsf file; its other series left out. Each forecast moved from the day it was issued to the day it is for. Columns renamed. Days without a figure left empty. The sample keeps one station.
- Full version: the same nine columns for all 422 stations, with the station
  number above each column. `load("rain", full=True)["86282"]` is Melbourne
  Airport. The download is 26 MB.

## River flow

`river-flow.csv`: the daily mean flow of the Saugeen River at Walkerton,
Ontario, in cubic meters per second, 10,957 days from 1 January 1950 to
31 December 1979.

- Source: Saugeen River Flow (SaugeenDay) Dataset, Monash Time Series Forecasting Archive, on Zenodo,
  <https://zenodo.org/records/4656058>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). Saugeen River Flow (SaugeenDay) Dataset (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656058. The archive took the series from the R package deseasonalize.
- Changes: Converted from the archive's .tsf file to a table with one row per day. The sample keeps 1950 to 1979.
- Full version: the same series from 1 January 1915, 23,741 days.

## Site views

`site-views.csv`: the page views of the whole OTexts.com site, which publishes
textbooks online, on each day from 1 January 2018 to 10 April 2022, 1,561
days. The package says the counts are as recorded by Google Analytics. The
sample is the whole series.

- Source: otexts_views, R package fpp3, as a CSV file in Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/otexts_views.html>
- License: [GPL-3 (R package fpp3)](https://github.com/robjhyndman/fpp3-package/blob/master/DESCRIPTION), the license
  of the package the table ships in.
- Credit: Hyndman, R. J., and others. fpp3: Data for "Forecasting: Principles and Practice" (3rd Edition) (R package), dataset otexts_views. The package describes the series as daily page views on the OTexts website (https://OTexts.com/) as recorded by Google Analytics. Distributed by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The row-number column dropped; the columns named date and views.

## Solar farm

`solar-farm.csv`: the average output of one solar farm in megawatts in each
half hour of 339 days, 16,272 half hours. The file has one row per day and one
column per half hour; the loader returns one row per half hour. The readings
were made every four seconds, and the archive took them from the Australian
Energy Market Operator. It does not name the farm.

The archive gives the series one start time, 1 August 2019, and no other
timestamps, so the loader places each reading four seconds after the one
before. That clock is not the farm's local time. On the second day the series
jumps from late morning straight into night, so readings are missing there,
and from then on the farm's daylight falls around midnight on the archive's
clock. Over the year the daylight also comes about an hour earlier on that
clock than the seasons explain, which suggests more readings are missing along
the way. The sample leaves out the first two and a half days and starts each
day at 12:00 on the archive's clock, which is in the farm's night.

- Source: Solar Power Dataset (4 Seconds Observations), Monash Time Series
  Forecasting Archive, on Zenodo, <https://zenodo.org/records/4656027>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the archive's
  license for its file. The readings are the Australian Energy Market
  Operator's. Its Copyright Permissions page is reported to give a general
  permission to use its material for any purpose with accurate attribution to
  the operator as author.
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Abolghasemi, M., Hyndman, R., & Montero-Manso, P. (2020). Solar Power Dataset (4 Seconds Observations) (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656027. The archive downloaded the readings from the Australian Energy Market Operator (AEMO), https://aemo.com.au/.
- Changes: Converted from the archive's .tsf file. The archive gives one start time and no other timestamps, so each reading is placed four seconds after the one before it; the result is the archive's clock, not the farm's local time. The sample averages the readings over half hours, rounds to one decimal, leaves out the first two and a half days, where readings are missing, and the last part day, and holds one row per day and one column per half hour, with each day starting at 12:00 on the archive's clock; the loader returns one row per half hour.
- Full version: the output every four seconds, 7,397,222 readings in one
  column on the archive's clock.

## Southern Oscillation

`southern-oscillation.csv`: the Southern Oscillation Index for every month
from January 1900 to December 2005, 1,272 months. The index compares the air
pressure at Tahiti with the air pressure at Darwin; months of low values go
with El Niño and months of high values with La Niña. The sample is the whole
series.

- Source: bomsoi, R package DAAG, as a CSV file in Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomsoi.html>
- License: [GPL-3 (R package DAAG)](https://github.com/cran/DAAG/blob/master/DESCRIPTION), the license
  of the package the table ships in. The package gives the web pages of the
  Australian Bureau of Meteorology as the source of its figures; the Bureau's
  own terms for them are not stated in the package.
- Credit: Maindonald, J. H., & Braun, W. J. DAAG: Data Analysis and Graphics Data and Functions (R package), dataset bomsoi. The package gives the web pages of the Australian Bureau of Meteorology as the source of the figures. Distributed by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The twelve monthly columns of each year put in one column in date order, named soi; the yearly average and the rainfall columns left out.

## Souvenir sales

`souvenir-sales.csv`: the monthly sales of a souvenir shop on the wharf of a
beach resort town in Queensland, Australia, from January 1987 to December
1993, 84 months. The package does not say what unit the sales are in. The
sample is the whole series.

- Source: souvenirs, R package fpp3, as a CSV file in Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/souvenirs.html>
- License: [GPL-3 (R package fpp3)](https://github.com/robjhyndman/fpp3-package/blob/master/DESCRIPTION), the license
  of the package the table ships in. The package gives a textbook exercise as
  its source and does not name the shop.
- Credit: Hyndman, R. J., and others. fpp3: Data for "Forecasting: Principles and Practice" (3rd Edition) (R package), dataset souvenirs. The package describes the series as monthly sales for a souvenir shop on the wharf at a beach resort town in Queensland, Australia, and gives Makridakis, Wheelwright and Hyndman (1998), Forecasting: Methods and Applications, exercise 5.8, as its source. Distributed by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The row-number column dropped; the columns named month and sales, with the month written as year and month number.

## Share index

`sp500.csv`: the level of the S&P 500 share index at the close of the last
trading day of each week, 3,444 weeks ending on Fridays from 6 January 1950
to 1 January 2016. The index leaves out dividends.

- Source: sp500, R package gt, as a CSV file in Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/doc/gt/sp500.html>
- License: [MIT (R package gt)](https://github.com/rstudio/gt/blob/master/LICENSE.md), the license
  of the package the table ships in. The package does not say where its
  figures were collected.
- Credit: Iannone, R., Cheng, J., Schloerke, B., and others. gt: Easily Create Presentation-Ready Display Tables (R package, Posit Software, PBC), dataset sp500. Distributed by Rdatasets (https://github.com/vincentarelbundock/Rdatasets). The index is calculated by S&P Dow Jones Indices.
- Changes: Rows put in date order; the row-number column and the adjusted close, which equals the close on every day, dropped. The sample keeps the close of the last trading day of each week.
- Full version: open, high, low, close and volume for each of the 16,607
  trading days from 3 January 1950 to 31 December 2015, on a calendar of
  business days. A row is empty where the market was closed. Before 1962 the
  source gives the close in all four price columns.

## Sunspots

`sunspots.csv`: the yearly sunspot number, 289 years from 1700 to 1988. The
sample is the whole series. These are the numbers as published up to 1988;
the keepers of the series, WDC-SILSO at the Royal Observatory of Belgium, have
since revised and extended it.

- Source: sunspot.year, R datasets package, as a CSV file in Rdatasets,
  <https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/sunspot.year.html>
- License: [GPL-2 | GPL-3 (R datasets package)](https://stat.ethz.ch/R-manual/R-devel/library/datasets/DESCRIPTION)
- Credit: Tong, H. (1996). Non-Linear Time Series. Clarendon Press, Oxford, p. 471. Distributed with R's datasets package and by Rdatasets (https://github.com/vincentarelbundock/Rdatasets).
- Changes: The row-number column dropped, and the columns named year and sunspots.

## Wikipedia views

`wikipedia-views.csv`: how many times a day each of 136 Wikipedia pages was
viewed, 803 days from 1 July 2015 to 10 September 2017. Each column is one
page, under the archive's number for it (`T128`, `T5019` and so on). The
archive does not carry the names of the pages. The competition the counts were
collected for counted a page separately by device and for search engine
robots, so two columns of the archive can be the same page counted in
different ways, and some columns count robots; the numbering does not say
which.

- Source: Kaggle Wikipedia Web Traffic Daily Dataset (with Missing Values), Monash Time Series
  Forecasting Archive, on Zenodo, <https://zenodo.org/records/4656080>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the archive's
  license for its file. The counts themselves are the Wikimedia Foundation's
  page view statistics, which the Foundation releases under CC0.
- Credit: Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). Kaggle Wikipedia Web Traffic Daily Dataset (with Missing Values) (Version 3) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656080. The archive took the series from the Kaggle competition Web Traffic Time Series Forecasting (Google, 2017); the counts are the Wikimedia Foundation's page view statistics.
- Changes: Pages taken from the archive's .tsf file, one column per page under the archive's series number; the archive does not carry the names of the pages. Kept: every tenth page among those with a count for every day and a median of at least 100 views a day. No count filled in or changed. The sample keeps every fiftieth of those pages.
- Full version: 6,766 pages in the same layout, every tenth page among the
  67,656 that have a count for every day and a median of at least 100 views a
  day. The source file is a 145 MB download.

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
