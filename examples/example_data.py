"""Data for the example notebooks.

Every dataset has a small sample committed in ``examples/data``, so the
notebooks run offline, and a full version built on request from the public
source. ``examples/data/datasets.json`` records where each dataset comes from,
its license, what was changed and the checksum of every file.

    from example_data import load, about

    sales = load("store-sales")                  # the committed sample
    everything = load("store-sales", full=True)  # downloads the source once
    print(about("store-sales"))                  # source, license and credit

Every table comes back with a regular date index (from half-hourly to yearly),
which is what TimeGym needs to hand a forecaster the dates it must forecast.

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


def _archive_series(source, member):
    """Every series in one of the archive's ``.tsf`` files, by name, with its dates as the index."""
    steps = {"half_hourly": "30min", "daily": "D", "monthly": "MS"}
    with zipfile.ZipFile(source) as archive:
        text = archive.read(member).decode("utf-8")
    step, series = None, {}
    for line in text.splitlines():
        if line.startswith("@frequency"):
            step = steps[line.split()[1]]
        if not line or line[0] in "#@":
            continue
        name, start, values = line.split(":", 2)
        # The archive writes a missing value as "?".
        values = pd.to_numeric(pd.Series(values.split(",")), errors="coerce")
        values.index = pd.date_range(start.split(" ")[0], periods=len(values), freq=step)
        series[name] = values
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


def _write_spare_parts(table, path):
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
        "write": _write_spare_parts,
        "sample": _sample_spare_parts,
    },
    "electricity-demand": {
        "read": _read_electricity,
        "build": _build_electricity,
        "write": _write_electricity,
        "sample": lambda table: table,
    },
    "river-flow": {
        "read": _read_daily,
        "build": _build_river_flow,
        "write": _write_daily,
        "sample": _sample_river_flow,
    },
    "sunspots": {
        "read": _read_sunspots,
        "build": _build_sunspots,
        "write": _write_sunspots,
        "sample": lambda table: table,
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
            if table.index.freq is None or table.isna().any().any():
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
