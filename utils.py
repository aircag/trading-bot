import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import ta


BASE_URL = "https://api.mexc.co"
MAX_KLINE_LIMIT = 1000

INTERVAL_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
}

logger = logging.getLogger("mexc-utils")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _log(message: str) -> None:
    """Central logging helper so Streamlit console stays readable."""
    logger.info(message)


def _safe_get(url: str, params: dict | None = None, retries: int = 3, delay: float = 0.5):
    """Retry wrapper for GET requests against public MEXC endpoints."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            _log(f"Request failed ({attempt}/{retries}): {exc}")
            time.sleep(delay)
    return None


def _to_utc_millis(value: Any) -> int:
    """Convert many timestamp formats to UTC milliseconds."""
    if value is None:
        raise ValueError("Timestamp value cannot be None")

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return int(ts.value // 1_000_000)

    if isinstance(value, datetime):
        ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        return int(ts.timestamp() * 1000)

    if isinstance(value, str):
        return _to_utc_millis(pd.Timestamp(value))

    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def get_klines(
    symbol: str,
    interval: str = "30m",
    limit: int = 120,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Single REST call to fetch klines."""
    params: dict[str, Any] = {
        "symbol": f"{symbol.upper()}USDT",
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    resp = _safe_get(f"{BASE_URL}/api/v3/klines", params=params)
    if not resp:
        return pd.DataFrame()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
    ]
    df = pd.DataFrame(resp.json(), columns=cols).astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def get_historical_klines(
    symbol: str,
    interval: str = "30m",
    days: int = 180,
    start: datetime | pd.Timestamp | str | None = None,
    end: datetime | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Paginate through the public kline endpoint and return a clean DataFrame."""
    minutes = INTERVAL_MINUTES.get(interval)
    if minutes is None:
        raise ValueError(f"Unsupported interval: {interval}")

    millis_per_candle = minutes * 60 * 1000
    now_ms = int(time.time() * 1000)
    custom_range = start is not None and end is not None

    if custom_range:
        start_time = _to_utc_millis(start)
        end_time = min(_to_utc_millis(end), now_ms)
        if start_time >= end_time:
            raise ValueError("Start time must be before end time.")
        _log(
            f"Fetching {symbol.upper()} {interval} candles "
            f"{pd.to_datetime(start_time, unit='ms', utc=True)} → {pd.to_datetime(end_time, unit='ms', utc=True)}"
        )
    else:
        lookback_days = max(days, 1)
        end_time = now_ms
        start_time = end_time - lookback_days * 24 * 60 * 60 * 1000
        _log(f"Fetching {symbol.upper()} {interval} candles for {lookback_days} days")

    frames: list[pd.DataFrame] = []
    cursor = start_time

    while cursor < end_time:
        remaining_ms = end_time - cursor
        remaining_candles = max(1, math.ceil(remaining_ms / millis_per_candle))
        limit = min(MAX_KLINE_LIMIT, remaining_candles)
        batch_end = min(cursor + limit * millis_per_candle, end_time)

        df = get_klines(
            symbol,
            interval=interval,
            limit=limit,
            start_time=cursor,
            end_time=batch_end,
        )

        if df.empty:
            _log(
                f"No candles returned for window "
                f"{datetime.utcfromtimestamp(cursor/1000)} → {datetime.utcfromtimestamp(batch_end/1000)}"
            )
            cursor = batch_end
            continue

        frames.append(df)
        last_open = df.iloc[-1]["open_time"]
        next_cursor = int(last_open.timestamp() * 1000) + millis_per_candle
        cursor = max(next_cursor, cursor + millis_per_candle)

        if cursor >= end_time:
            break

        time.sleep(0.15)  # stay polite with the public API

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    start_cutoff = pd.Timestamp.fromtimestamp(start_time / 1000, tz="UTC")
    end_cutoff = pd.Timestamp.fromtimestamp(end_time / 1000, tz="UTC")
    mask = (df_all["open_time"] >= start_cutoff) & (df_all["open_time"] <= end_cutoff)
    filtered = df_all.loc[mask].reset_index(drop=True)
    return filtered


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach RSI, Bollinger lower band and a 20-period volume MA."""
    if df.empty:
        return df

    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_lower"] = bb.bollinger_lband()
    df["vol_ma20"] = ta.trend.SMAIndicator(df["volume"], 20).sma_indicator()
    df["is_green"] = df["close"] > df["open"]
    return df


__all__ = ["INTERVAL_MINUTES", "add_indicators", "get_historical_klines"]
