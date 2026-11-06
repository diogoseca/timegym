# Market data

An optional, opt-in helper that fetches and caches daily stock history:
`get_stock` from `timegym.data.real`.

`timegym.data.real.get_stock(ticker, columns="Close", *, auto_adjust=True,
cache_path=None, refresh=False)` is a public opt-in network helper. Install
`timegym[market-data]` for a fetch. It requests one ticker's maximum daily history
through [yfinance's history API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html).
It always returns a new pandas DataFrame, including for a single column, in the
requested column order with the provider's timezone and dates (nanosecond index
resolution). Missing observations are preserved; empty history, unavailable
columns, and duplicate or unordered dates raise `ValueError`.

`auto_adjust=True`, the default, requests adjusted OHLC including `Close`, so
splits and dividends do not appear as jumps; `False` keeps the provider's
unadjusted fields. Choose and record this setting before defining the
forecasting target. Select `result["Close"]` for a Series.

`cache_path` names a JSON snapshot file containing full history, ticker and
adjustment metadata. A matching existing snapshot is read without importing
yfinance or making a network request. It can serve different column selections.
Malformed snapshots or different ticker/adjustment settings raise; they never
trigger an implicit refetch. `refresh=True` fetches again and atomically replaces
the snapshot only after validation. Fetch errors propagate, and an unsuccessful
refresh leaves the previous snapshot intact. No cache is written when the path
is omitted. JSON preserves numeric values to 15 decimal places; comparison of
floating-point values across a cache round trip should allow rounding tolerance.

```python
from timegym.data.real import get_stock

prices = get_stock("MSFT", auto_adjust=True, cache_path="data/msft-adjusted.json")
```

The first call above uses the network unless the snapshot already exists.
Archive the snapshot and adjustment setting with an experiment: provider history
can be revised, and today's adjusted prices are not point-in-time data. This
helper supplies observations, not a known conditional reference. The maintained
notebooks instead use the bundled AirPassengers CSV and run offline on CPU.
