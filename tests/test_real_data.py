import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from timegym.data.real import get_stock


@pytest.fixture
def stock_history(monkeypatch):
    history = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0],
            "Close": [102.0, 101.0, 104.0],
            "Volume": [1000, 1200, 900],
        },
        index=pd.date_range(
            "2024-03-08", periods=3, tz="America/New_York", name="Date"
        ),
    )
    calls = []

    def ticker(symbol):
        def fetch(**kwargs):
            calls.append((symbol, kwargs))
            result = history.copy()
            if kwargs["auto_adjust"]:
                result[["Open", "Close"]] *= 0.5
            return result

        return SimpleNamespace(history=fetch)

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker))
    return history, calls


@pytest.mark.parametrize("adjusted", [False, True])
def test_stock_selection_is_always_a_frame_and_adjustment_is_explicit(
    stock_history, adjusted
):
    history, calls = stock_history
    single = get_stock(" msft ", auto_adjust=adjusted)
    multiple = get_stock("MSFT", ["Volume", "Close"], auto_adjust=adjusted)
    assert isinstance(single, pd.DataFrame)
    assert single.columns.tolist() == ["Close"]
    assert multiple.columns.tolist() == ["Volume", "Close"]
    assert single["Close"].tolist() == [
        value * (0.5 if adjusted else 1) for value in history.Close
    ]
    assert calls == [("MSFT", {"period": "max", "auto_adjust": adjusted})] * 2
    single.iloc[0, 0] = 0
    assert history.Close.iloc[0] == 102


def test_cache_reuses_full_history_without_optional_dependency(
    stock_history, monkeypatch, tmp_path
):
    history, calls = stock_history
    path = tmp_path / "nested" / "msft.json"
    first = get_stock("MSFT", ["Volume", "Close"], cache_path=path, auto_adjust=True)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    cached = get_stock("MSFT", ["Volume", "Close"], cache_path=path, auto_adjust=True)
    pd.testing.assert_frame_equal(first, cached, check_freq=False)
    assert str(cached.index.tz) == "America/New_York"
    assert get_stock(
        "MSFT", "Open", cache_path=path, auto_adjust=True
    ).Open.tolist() == [50, 51, 50.5]
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"ticker": "OTHER"}, {"auto_adjust": True}])
def test_cache_mismatch_never_fetches(stock_history, tmp_path, kwargs):
    _, calls = stock_history
    path = tmp_path / "stock.json"
    get_stock("MSFT", cache_path=path)
    options = {"ticker": "MSFT", "cache_path": path, **kwargs}
    with pytest.raises(ValueError, match="Invalid stock cache.*does not match"):
        get_stock(**options)
    assert len(calls) == 1


@pytest.mark.parametrize("text", ["not JSON", "{}", "[]", "null"])
def test_malformed_cache_never_fetches(stock_history, tmp_path, text):
    _, calls = stock_history
    path = tmp_path / "stock.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="Invalid stock cache"):
        get_stock("MSFT", cache_path=path)
    assert calls == []


def test_refresh_explicitly_replaces_snapshot(stock_history, tmp_path):
    history, calls = stock_history
    path = tmp_path / "stock.json"
    get_stock("MSFT", cache_path=path)
    history.loc[:, "Close"] += 10
    refreshed = get_stock("MSFT", cache_path=path, refresh=True, auto_adjust=True)
    assert refreshed.Close.tolist() == [56, 55.5, 57]
    assert json.loads(path.read_text())["metadata"]["auto_adjust"] is True
    assert len(calls) == 2


def test_failed_refresh_keeps_previous_snapshot(stock_history, monkeypatch, tmp_path):
    path = tmp_path / "stock.json"
    get_stock("MSFT", cache_path=path)
    previous = path.read_bytes()

    def unavailable(**kwargs):
        raise ConnectionError("provider unavailable")

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(Ticker=lambda ticker: SimpleNamespace(history=unavailable)),
    )
    with pytest.raises(ConnectionError, match="provider unavailable"):
        get_stock("MSFT", cache_path=path, refresh=True)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_empty_provider_history_does_not_create_cache(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(
            Ticker=lambda ticker: SimpleNamespace(
                history=lambda **kwargs: pd.DataFrame()
            )
        ),
    )
    path = tmp_path / "stock.json"
    with pytest.raises(ValueError, match="nonempty"):
        get_stock("MSFT", cache_path=path)
    assert not path.exists()


def test_stock_missing_columns_do_not_create_cache(stock_history, tmp_path):
    path = tmp_path / "stock.json"
    with pytest.raises(ValueError, match="missing requested columns.*Adj Close"):
        get_stock("MSFT", "Adj Close", cache_path=path, auto_adjust=True)
    assert not path.exists()


def test_fetch_requires_market_data_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"timegym\[market-data\]"):
        get_stock("MSFT")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"ticker": " "}, ValueError),
        ({"ticker": None}, ValueError),
        ({"columns": []}, ValueError),
        ({"columns": ["Close", "Close"]}, ValueError),
        ({"columns": [1]}, ValueError),
        ({"columns": None}, TypeError),
        ({"auto_adjust": 1}, TypeError),
        ({"refresh": "yes"}, TypeError),
    ],
)
def test_stock_rejects_invalid_arguments_before_fetch(stock_history, kwargs, error):
    _, calls = stock_history
    with pytest.raises(error):
        get_stock(**{"ticker": "MSFT", **kwargs})
    assert calls == []
