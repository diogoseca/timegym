"""Data for the example notebooks.

Every dataset has a small sample committed in ``examples/data``, so the
notebooks run offline, and a full version built on request from the public
source. ``examples/data/datasets.json`` records where each dataset comes from,
its license, what was changed and the checksum of every file.

    from example_data import load, about

    sales = load("store-sales")                  # the committed sample
    everything = load("store-sales", full=True)  # downloads the source once
    print(about("store-sales"))                  # source, license and credit

Every table comes back indexed by date. All but one have a regular step (from
half-hourly to yearly), which TimeGym's built-in forecasters need to work out
the dates they must forecast. ``bank-calls`` follows a call center's opening
hours instead: it has no rows for nights, weekends and holidays.

From a terminal, ``python examples/example_data.py --check`` verifies the
committed samples and ``--rebuild NAME`` writes a sample again from its source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
RECORDS = DATA / "datasets.json"


def _records():
    return json.loads(RECORDS.read_text(encoding="utf-8"))


def names():
    """The names ``load`` accepts."""
    return list(_records())


def record(name):
    """The dataset's full record from ``datasets.json``, as a dictionary."""
    records = _records()
    if name not in records:
        raise KeyError(f"No dataset named {name!r}. Choose from: {', '.join(records)}")
    return records[name]


def about(name):
    """A few lines on the dataset: what it holds, its source, license and credit."""
    entry = record(name)
    source, sample = entry["source"], entry["sample"]
    lines = [
        entry["title"],
        f"  Sample:  {sample['holds']}",
        f"  Source:  {source['name']}, {source['publisher']} ({source['page']})",
        f"  License: {entry['license']['name']} ({entry['license']['url']})",
        f"  Credit:  {entry['credit']}",
        f"  Changes: {entry['changes']}",
    ]
    return "\n".join(lines)


def load(name, *, full=False):
    """Return the dataset as a table with a regular date index.

    By default this reads the small sample committed in ``examples/data``.
    ``full=True`` downloads the public source once, checks it against the
    recorded checksum and builds the whole table; later calls read a cached
    copy. Where the sample already is the whole dataset, both give the same
    table.
    """
    entry = record(name)
    reader = _DATASETS[name]["read"]
    if not full or entry["sample"].get("complete", False):
        return reader(DATA / entry["sample"]["file"])
    return reader(_full_table(name))


# Where downloads and full tables are kept


def cache_folder():
    """The folder for downloads: ``TIMEGYM_EXAMPLES_CACHE`` or ``~/.cache/timegym-examples``."""
    chosen = os.environ.get("TIMEGYM_EXAMPLES_CACHE")
    return Path(chosen) if chosen else Path.home() / ".cache" / "timegym-examples"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file(name, given=None):
    """Path to the dataset's source file, downloaded if needed and checked against its checksum."""
    source = record(name)["source"]
    path = Path(given) if given else cache_folder() / name / source["file"]
    if not path.exists():
        # Two datasets can come from one source file: use a copy that is already there.
        copies = [copy for copy in cache_folder().glob(f"*/{source['file']}") if _sha256(copy) == source["sha256"]]
        path = copies[0] if copies else path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        megabytes = source["bytes"] / 1e6
        print(f"Downloading {source['file']} ({megabytes:.1f} MB) from {source['download']}")
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(source["download"], timeout=120) as reply:
            with open(partial, "wb") as handle:
                for block in iter(lambda: reply.read(1 << 20), b""):
                    handle.write(block)
        partial.replace(path)
    found = _sha256(path)
    if found != source["sha256"]:
        raise ValueError(
            f"{path} does not match the recorded checksum for {name!r}. The source may have "
            f"changed, or the download was cut short; delete the file to download it again.\n"
            f"  recorded {source['sha256']}\n  found    {found}"
        )
    return path


def _full_table(name, given=None):
    """Path to the cached full table, built from the source on first use."""
    path = cache_folder() / name / f"{name}-full.csv"
    stamp = path.with_name(path.name + ".source")
    source = record(name)["source"]["sha256"]
    if not path.exists() or not stamp.exists() or stamp.read_text().strip() != source:
        table = _DATASETS[name]["build"](_source_file(name, given))
        path.parent.mkdir(parents=True, exist_ok=True)
        _DATASETS[name]["write"](table, path)
        stamp.write_text(source + "\n")
    return path


# Weekly units sold by an online shop


def _read_store_sales(path):
    """Weekly units per product: one column per product name, weeks starting on Monday."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        codes = next(rows)[1:]
        products = next(rows)[1:]
    table = pd.read_csv(path, skiprows=2, header=None, index_col=0, parse_dates=True)
    labels = _unique_labels(products, codes)
    table.columns = pd.Index(labels, name="product")
    table.index = pd.DatetimeIndex(table.index, freq="W-MON", name="week")
    table.attrs["codes"] = dict(zip(labels, codes))
    return table


def _unique_labels(products, codes):
    """Product names, with the stock code added where two products share a name."""
    seen = pd.Series(products).value_counts()
    return [
        f"{product} ({code})" if seen[product] > 1 else product
        for product, code in zip(products, codes)
    ]


def _write_store_sales(table, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        rows = csv.writer(handle, lineterminator="\n")
        rows.writerow(["stock_code", *table.columns])
        rows.writerow(["week", *(table.attrs["products"][code] for code in table.columns)])
        for week, units in zip(table.index, table.to_numpy()):
            rows.writerow([week.strftime("%Y-%m-%d"), *(int(value) for value in units)])


def _build_store_sales(source):
    """Weekly units for every product, from the two-year order log (needs ``openpyxl``)."""
    with zipfile.ZipFile(source) as archive:
        workbook = io.BytesIO(archive.read("online_retail_II.xlsx"))
    sheets = pd.read_excel(workbook, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    first, second = sheets["Year 2009-2010"], sheets["Year 2010-2011"]
    # The second sheet repeats the first sheet's last nine days; keep each order line once.
    lines = pd.concat(
        [first, second[second["InvoiceDate"] > first["InvoiceDate"].max()]], ignore_index=True
    )
    lines["code"] = lines["StockCode"].str.strip().str.upper()
    # Product codes are five digits and sometimes letters; the rest are postage, fees and adjustments.
    lines = lines[lines["code"].str.fullmatch(r"\d{5}[A-Z]*") & (lines["Price"] > 0)]
    lines = lines.assign(week=lines["InvoiceDate"].dt.to_period("W-SUN").dt.start_time)
    # Cancelled orders carry negative quantities, so the sum is what was sold and kept.
    units = lines.groupby(["week", "code"])["Quantity"].sum().unstack(fill_value=0)
    weeks = pd.date_range(units.index.min(), units.index.max(), freq="W-MON", name="week")
    # Weeks with no orders (the shop closes over Christmas) count as zero; the first and
    # last weeks are cut short by the start and end of the log, so they are dropped.
    units = units.reindex(weeks, fill_value=0).iloc[1:-1].clip(lower=0).astype(int)
    units = units.loc[:, units.sum() > 0]
    sold = lines[lines["Quantity"] > 0]
    described = sold.dropna(subset=["Description"])
    described = described.assign(
        product=described["Description"].astype(str).str.split().str.join(" ")
        .str.rstrip(" .").str.capitalize()
    )
    products = (
        described.groupby(["code", "product"]).size().rename("lines").reset_index()
        .sort_values(["code", "lines", "product"], ascending=[True, False, True])
        .drop_duplicates("code").set_index("code")["product"]
    )
    order_lines = sold.groupby("code").size()
    units = units.loc[:, [code for code in units.columns if code in products.index]]
    units.attrs["products"] = products.reindex(units.columns).to_dict()
    units.attrs["order_lines"] = order_lines.reindex(units.columns).to_dict()
    return units


def _sample_store_sales(table):
    """The 30 products with the most order lines among those sold in at least 100 of the weeks."""
    weeks_sold = (table > 0).sum()
    order_lines = pd.Series(table.attrs["order_lines"])
    steady = order_lines[weeks_sold[order_lines.index] >= 100]
    ranked = steady.rename_axis("code").reset_index(name="lines")
    ranked = ranked.sort_values(["lines", "code"], ascending=[False, True])
    sample = table[list(ranked["code"].head(30))]
    sample.attrs = dict(table.attrs)
    return sample


# Files from the Monash Time Series Forecasting Archive


def _archive_series(source, member, keep=None):
    """Series from one of the archive's ``.tsf`` files, each with its dates as the index.

    A series is filed under its name, or under its other labels (such as station and
    measurement) when the file gives any. ``keep`` chooses series by those labels.
    """
    steps = {"half_hourly": "30min", "hourly": "h", "daily": "D", "monthly": "MS"}
    step, labels, series = None, [], {}
    with zipfile.ZipFile(source) as archive, archive.open(member) as lines:
        for line in io.TextIOWrapper(lines, encoding="utf-8"):
            line = line.strip()
            if line.startswith("@attribute"):
                labels.append(line.split()[1])
            elif line.startswith("@frequency"):
                step = steps[line.split()[1]]
            if not line or line[0] in "#@":
                continue
            *fields, values = line.split(":", len(labels))
            record = dict(zip(labels, fields))
            # A start is written as "2017-01-01 14-00-00": a day, and an hour that counts
            # for series with more than one value a day.
            name, (day, _, clock) = record.pop("series_name"), record.pop("start_timestamp").partition(" ")
            start = pd.Timestamp(day) + pd.Timedelta(hours=int(clock[:2]) if step in ("30min", "h") else 0)
            if keep is not None and not keep(record):
                continue
            # The archive writes a missing value as "?".
            values = pd.to_numeric(pd.Series(values.split(",")), errors="coerce")
            values.index = pd.date_range(start, periods=len(values), freq=step)
            series[tuple(record.values()) if record else name] = values
    return series


def _read_daily(path):
    table = pd.read_csv(path, index_col="date", parse_dates=True)
    table.index = pd.DatetimeIndex(table.index, freq="D", name="date")
    return table


def _read_monthly(path):
    table = pd.read_csv(path)
    table.index = pd.PeriodIndex(table.pop("month"), freq="M", name="month")
    return table


def _write_daily(table, path):
    table.to_csv(path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n")


# Daily Bitcoin price and network figures


def _build_bitcoin(source):
    """All 18 daily series, one column each; a series is empty before its first day."""
    table = pd.DataFrame(_archive_series(source, "bitcoin_dataset_without_missing_values.tsf"))
    return table[["price", *(name for name in table.columns if name != "price")]]


def _sample_bitcoin(table):
    """The price only, from its first day."""
    return table[["price"]].dropna()


# Daily flow of a river


def _build_river_flow(source):
    """The daily mean flow from 1915, in cubic meters per second."""
    flow = _archive_series(source, "saugeenday_dataset.tsf")["T1"]
    return pd.DataFrame({"flow": flow})


def _sample_river_flow(table):
    """The last 30 years, 1950 to 1979."""
    return table.loc["1950-01-01":]


# Monthly sales of spare parts


def _write_monthly_counts(table, path):
    # Whole units; a month the source has no figure for stays empty.
    table.astype("Int64").to_csv(path, index_label="month", lineterminator="\n")


def _build_spare_parts(source):
    """Units sold per month for every part; a part's months without a figure are empty."""
    table = pd.DataFrame(_archive_series(source, "car_parts_dataset_with_missing_values.tsf"))
    table.index = table.index.to_period("M").rename("month")
    return table.rename_axis(columns="part")


def _sample_spare_parts(table):
    """Every eighth part among those with a figure for every month."""
    complete = table.loc[:, table.notna().all()]
    return complete.iloc[:, ::8].astype(int)


# Monthly patient counts for medical products


def _build_hospital(source):
    """Patients per month for every product."""
    table = pd.DataFrame(_archive_series(source, "hospital_dataset.tsf"))
    table.index = table.index.to_period("M").rename("month")
    return table.rename_axis(columns="product")


def _sample_hospital(table):
    """Every second product among those with at least 100 patients a month on average."""
    busy = table.loc[:, table.mean() >= 100]
    return busy.iloc[:, ::2].astype(int)


# Daily rain and the official chance of rain

_RAIN_STATION = "66062"   # Sydney (Observatory Hill)
_RAIN_COLUMNS = {"PRCP_SUM": "rain_mm", "T_MAX": "max_temp", "T_MIN": "min_temp"}
_RAIN_FORECASTS = {
    0: "bureau_same_day", 1: "bureau_1_day_before", 2: "bureau_2_days_before",
    3: "bureau_3_days_before", 4: "bureau_4_days_before", 5: "bureau_5_days_before",
}


def _read_rain(path):
    """One station's table, or every station's with the station number above each column."""
    with open(path, encoding="utf-8") as handle:
        every_station = handle.readline().startswith("station,")
    table = pd.read_csv(path, header=[0, 1] if every_station else 0, index_col=0, parse_dates=True)
    table.index = pd.DatetimeIndex(table.index, freq="D", name="date")
    return table


def _write_rain(table, path):
    table.to_csv(path, index_label="date" if table.columns.nlevels == 1 else None,
                 date_format="%Y-%m-%d", lineterminator="\n")


def _build_rain(source):
    """Rain, temperature and the Bureau's chance of rain for all 422 stations.

    The archive files each forecast under the day it was issued. Here it is moved to the
    day it is for, so a row holds what happened and what had been forecast for it.
    """
    wanted = {**_RAIN_COLUMNS, **{f"fcst_{days}_DailyPoP": name for days, name in _RAIN_FORECASTS.items()}}
    series = _archive_series(source, "temperature_rain_dataset_with_missing_values.tsf",
                             keep=lambda labels: labels["obs_or_fcst"] in wanted)
    columns = {}
    for (station, kind), values in series.items():
        days_before = int(kind.split("_")[1]) if kind.startswith("fcst_") else 0
        columns[(station, wanted[kind])] = values.shift(days_before)
    table = pd.DataFrame(columns).rename_axis(index="date", columns=["station", None])
    order = [*_RAIN_COLUMNS.values(), *_RAIN_FORECASTS.values()]
    return table.reindex(columns=[(station, name) for station in table.columns.levels[0] for name in order])


def _sample_rain(table):
    """One station: Sydney (Observatory Hill)."""
    return table[_RAIN_STATION]


# Daily highest temperature in eight cities, from the same source as the rain table

_CITY_STATIONS = {
    "Sydney": "66062", "Melbourne": "86282", "Adelaide": "23034", "Canberra": "70351",
    "Hobart": "94029", "Perth": "9021", "Brisbane": "40842", "Darwin": "14015",
}


def _build_city_temperatures(source):
    """The highest temperature of each day at all 422 stations, under the Bureau's station numbers."""
    series = _archive_series(source, "temperature_rain_dataset_with_missing_values.tsf",
                             keep=lambda labels: labels["obs_or_fcst"] == "T_MAX")
    table = pd.DataFrame({station: values for (station, _), values in series.items()})
    return table.rename_axis(index="date", columns="station")


def _sample_city_temperatures(table):
    """Eight cities, named."""
    return table[list(_CITY_STATIONS.values())].set_axis(list(_CITY_STATIONS), axis=1)


# Hourly fine particles in London's air


def _read_hourly(path):
    table = pd.read_csv(path, index_col="time", parse_dates=True)
    table.index = pd.DatetimeIndex(table.index, freq="h", name="time")
    return table


def _write_hourly(table, path):
    table.to_csv(path, index_label="time", date_format="%Y-%m-%d %H:%M", lineterminator="\n")


def _build_london_air(source):
    """Fine particles (PM2.5) for every hour at London's 24 monitors, empty where nothing was reported."""
    series = _archive_series(
        source, "kdd_cup_2018_dataset_with_missing_values.tsf",
        keep=lambda labels: labels["city"] == "London" and labels["air_quality_measurement"] == "PM2.5",
    )
    table = pd.DataFrame({station: values for (_, station, _), values in series.items()})
    return table.asfreq("h").rename_axis(index="time", columns="monitor")


def _sample_london_air(table):
    """The last five months."""
    return table.loc["2017-11-01":]


# Half-hourly electricity demand


def _read_electricity(path):
    """Demand for each half-hour, from a file with one row per day and one column per half-hour."""
    days = pd.read_csv(path, index_col="date", parse_dates=True)
    times = pd.date_range(days.index[0], periods=days.size, freq="30min", name="time")
    return pd.DataFrame({"demand": days.to_numpy().ravel()}, index=times)


def _write_electricity(table, path):
    days = pd.DataFrame(
        table["demand"].to_numpy().reshape(-1, 48),
        index=pd.Index(table.index[::48].strftime("%Y-%m-%d"), name="date"),
        columns=table.index[:48].strftime("%H:%M"),
    )
    days.to_csv(path, float_format="%.3f", lineterminator="\n")


def _build_electricity(source):
    """Demand in gigawatts for every half-hour of 2014, rounded to the megawatt."""
    demand = _archive_series(source, "elecdemand_dataset.tsf")["T1"]
    demand.index = demand.index.rename("time")
    return pd.DataFrame({"demand": demand.round(3)})


# Monthly airline passengers


def _write_airline(table, path):
    table.to_csv(path, index_label="month", lineterminator="\n")


def _build_airline(source):
    values = pd.read_csv(source)["value"].astype(int)
    months = pd.period_range("1949-01", periods=len(values), freq="M", name="month")
    return pd.DataFrame({"passengers": values.to_numpy()}, index=months)


# Yearly sunspot numbers


def _read_sunspots(path):
    table = pd.read_csv(path, dtype={"year": str})
    table.index = pd.PeriodIndex(table.pop("year"), freq="Y", name="year")
    return table


def _write_sunspots(table, path):
    table.to_csv(path, index_label="year", lineterminator="\n")


def _build_sunspots(source):
    values = pd.read_csv(source)
    years = pd.PeriodIndex(values["time"].astype(str), freq="Y", name="year")
    return pd.DataFrame({"sunspots": values["value"].to_numpy(float)}, index=years)


# The Southern Oscillation Index, month by month


def _build_southern_oscillation(source):
    """The twelve monthly columns of each year, put in one column in date order."""
    years = pd.read_csv(source)
    if not (years["Year"].diff().dropna() == 1).all():
        raise ValueError("the source skips a year, so its months cannot be put on a regular calendar")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = pd.period_range(f"{years['Year'].iloc[0]}-01", periods=12 * len(years), freq="M", name="month")
    return pd.DataFrame({"soi": years[names].to_numpy(float).ravel()}, index=months)


# The S&P 500 share index


def _read_sp500(path):
    """Weekly closes in the sample; every column by business day in the full table."""
    table = pd.read_csv(path, index_col="date", parse_dates=True)
    weekly = (table.index[1] - table.index[0]).days == 7
    table.index = pd.DatetimeIndex(table.index, freq="W-FRI" if weekly else "B", name="date")
    return table


def _build_sp500(source):
    """Open, high, low, close and volume by business day, empty on days the market was closed."""
    days = pd.read_csv(source, index_col="date", parse_dates=True).sort_index()
    return days[["open", "high", "low", "close", "volume"]].asfreq("B").rename_axis("date")


def _sample_sp500(table):
    """The last close of each week, with weeks ending on Friday."""
    return table[["close"]].resample("W-FRI").last()


# Calls to a bank's call center


def _read_opening_hours(path):
    """Rows only for the times the center was open, so the index has no regular step."""
    return pd.read_csv(path, index_col="time", parse_dates=True)


def _build_bank_calls(source):
    """Calls in every five minutes the center was open, by its own clock."""
    table = pd.read_csv(source, usecols=["DateTime", "Calls"])
    times = pd.to_datetime(table["DateTime"]).dt.tz_localize(None)
    return pd.DataFrame({"calls": table["Calls"].to_numpy()}, index=pd.DatetimeIndex(times, name="time"))


def _sample_bank_calls(table):
    """Calls in each half hour from 7:00 to 21:00."""
    calls = table.loc[table.index.hour < 21, "calls"]
    return calls.groupby(calls.index.floor("30min")).sum().to_frame()


# Daily views of Wikipedia pages


def _build_wikipedia_views(source):
    """Daily views of one page in ten, among the pages with a count for every day and a
    median of at least 100 views a day. The archive numbers the pages and does not name them."""
    pages, start, eligible = {}, None, 0
    member = "kaggle_web_traffic_dataset_with_missing_values.tsf"
    with zipfile.ZipFile(source) as archive, archive.open(member) as lines:
        for line in io.TextIOWrapper(lines, encoding="utf-8"):
            if not line.strip() or line[0] in "#@":
                continue
            name, first_day, counts = line.strip().split(":", 2)
            start = start or first_day
            # The archive writes a day without a count as "?". Every series starts on the same day.
            if "?" in counts or first_day != start:
                continue
            views = np.array(counts.split(","), dtype=float)
            if np.median(views) < 100:
                continue
            if eligible % 10 == 0:
                pages[name] = views.astype(int)
            eligible += 1
    days = pd.date_range(start.split()[0], periods=len(views), freq="D", name="date")
    return pd.DataFrame(pages, index=days).rename_axis(columns="page")


def _sample_wikipedia_views(table):
    """Every fiftieth page of the full table."""
    return table.iloc[:, ::50]


_DATASETS = {
    "store-sales": {
        "read": _read_store_sales,
        "build": _build_store_sales,
        "write": _write_store_sales,
        "sample": _sample_store_sales,
    },
    "bitcoin": {
        "read": _read_daily,
        "build": _build_bitcoin,
        "write": _write_daily,
        "sample": _sample_bitcoin,
    },
    "airline-passengers": {
        "read": _read_monthly,
        "build": _build_airline,
        "write": _write_airline,
        "sample": lambda table: table,
    },
    "spare-parts": {
        "read": _read_monthly,
        "build": _build_spare_parts,
        "write": _write_monthly_counts,
        "sample": _sample_spare_parts,
    },
    "bank-calls": {
        "read": _read_opening_hours,
        "build": _build_bank_calls,
        "write": _write_hourly,
        "sample": _sample_bank_calls,
    },
    "city-temperatures": {
        "read": _read_daily,
        "build": _build_city_temperatures,
        "write": _write_daily,
        "sample": _sample_city_temperatures,
    },
    "electricity-demand": {
        "read": _read_electricity,
        "build": _build_electricity,
        "write": _write_electricity,
        "sample": lambda table: table,
    },
    "hospital-patients": {
        "read": _read_monthly,
        "build": _build_hospital,
        "write": _write_monthly_counts,
        "sample": _sample_hospital,
    },
    "london-air": {
        "read": _read_hourly,
        "build": _build_london_air,
        "write": _write_hourly,
        "sample": _sample_london_air,
    },
    "rain": {
        "read": _read_rain,
        "build": _build_rain,
        "write": _write_rain,
        "sample": _sample_rain,
    },
    "river-flow": {
        "read": _read_daily,
        "build": _build_river_flow,
        "write": _write_daily,
        "sample": _sample_river_flow,
    },
    "southern-oscillation": {
        "read": _read_monthly,
        "build": _build_southern_oscillation,
        "write": _write_airline,
        "sample": lambda table: table,
    },
    "sp500": {
        "read": _read_sp500,
        "build": _build_sp500,
        "write": _write_daily,
        "sample": _sample_sp500,
    },
    "sunspots": {
        "read": _read_sunspots,
        "build": _build_sunspots,
        "write": _write_sunspots,
        "sample": lambda table: table,
    },
    "wikipedia-views": {
        "read": _read_daily,
        "build": _build_wikipedia_views,
        "write": _write_daily,
        "sample": _sample_wikipedia_views,
    },
}


# Checking and rebuilding the committed samples


def check():
    """Verify every committed sample; returns a list of problems, empty when all is well."""
    problems = []
    for name, entry in _records().items():
        path = DATA / entry["sample"]["file"]
        if name not in _DATASETS:
            problems.append(f"{name}: datasets.json lists it but example_data.py cannot read it")
        elif not path.exists():
            problems.append(f"{name}: {path.name} is missing")
        elif _sha256(path) != entry["sample"]["sha256"]:
            problems.append(f"{name}: {path.name} does not match its recorded checksum")
        else:
            table = load(name)
            gaps = table.isna().any().any() and not entry["sample"].get("gaps", False)
            uneven = table.index.freq is None and entry["sample"].get("regular", True)
            if uneven or gaps:
                problems.append(f"{name}: the sample needs a regular date index and no gaps")
    problems += [f"{name}: missing from datasets.json" for name in _DATASETS if name not in _records()]
    return problems


def rebuild(name, source=None):
    """Write the committed sample again from the source and record its checksum."""
    records = _records()
    table = _DATASETS[name]["build"](_source_file(name, source))
    path = DATA / records[name]["sample"]["file"]
    _DATASETS[name]["write"](_DATASETS[name]["sample"](table), path)
    records[name]["sample"]["sha256"] = _sha256(path)
    RECORDS.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check or rebuild the example data samples.")
    parser.add_argument("--check", action="store_true", help="verify the committed samples")
    parser.add_argument("--rebuild", metavar="NAME", help="write a sample again from its source")
    parser.add_argument("--source", metavar="FILE", help="a copy of the source file to use")
    args = parser.parse_args(argv)
    if args.rebuild:
        path = rebuild(args.rebuild, args.source)
        print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"{len(_records())} samples match their recorded checksums.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
