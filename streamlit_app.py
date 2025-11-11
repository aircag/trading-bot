import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from utils import (
    INTERVAL_MINUTES,
    add_indicators,
    get_historical_klines,
)


st.set_page_config(page_title="RSI + BB Signal Explorer", layout="wide")
st.title("RSI + Bollinger Band Signal Explorer")
st.caption("Scan MEXC data for oversold signals and simulate quick rebounds.")

CANDLE_INTERVAL = "15m"  # candle timeframe stays static for signal detection
BASE_INTERVAL_MINUTES = INTERVAL_MINUTES[CANDLE_INTERVAL]
STOP_LOSS_PCT = 0.01


def normalize_symbol(symbol: str) -> str:
    return symbol.replace("USDT", "").strip().upper()


def format_utc(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _parse_iso(ts: str | None):
    if not ts:
        return None
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_history(symbol: str,
                 days: int | None,
                 start_iso: str | None,
                 end_iso: str | None) -> pd.DataFrame:
    start_dt = _parse_iso(start_iso)
    end_dt = _parse_iso(end_iso)
    df = get_historical_klines(symbol,
                               interval=CANDLE_INTERVAL,
                               days=int(days) if days is not None else 0,
                               start=start_dt,
                               end=end_dt)
    return add_indicators(df)


with st.sidebar:
    st.header("Scan Settings")
    symbol_input = st.text_input("Symbol", value="XRP")
    interval = st.selectbox(
        "Hold Interval (exit horizon)",
        options=list(INTERVAL_MINUTES.keys()),
        index=list(INTERVAL_MINUTES.keys()).index("30m"),
    )
    lookback_mode = st.radio(
        "Lookback window",
        options=("Relative days", "Custom dates"),
        index=0,
    )
    days = None
    start_iso = None
    end_iso = None
    range_valid = True
    if lookback_mode == "Relative days":
        days = int(st.number_input("Lookback (days)", min_value=1, max_value=365, value=60, step=5))
    else:
        default_end = datetime.utcnow().date()
        default_start = default_end - timedelta(days=30)
        date_range = st.date_input(
            "Custom date range (UTC)",
            value=(default_start, default_end),
        )
        if isinstance(date_range, tuple):
            start_date = date_range[0]
            end_date = date_range[1] if len(date_range) > 1 else date_range[0]
        else:
            start_date = end_date = date_range

        if start_date is None or end_date is None:
            range_valid = False
        else:
            start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(days=1)
            if start_dt >= end_dt:
                range_valid = False
            else:
                start_iso = start_dt.isoformat()
                end_iso = end_dt.isoformat()

    rsi_threshold = st.number_input("RSI max", min_value=5.0, max_value=70.0, value=27.0, step=0.5)
    use_stop_loss = st.checkbox("Apply -1% protective stop", value=False,
                                help="When enabled, sells early if price drops 1% below entry before the timer exit.")
    run_scan = st.button("Find Signals", type="primary")

symbol = normalize_symbol(symbol_input)


def build_signals(df: pd.DataFrame, symbol_name: str, hold_interval: str, use_stop: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.dropna(subset=["rsi", "bb_lower"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    condition = (df["rsi"] <= rsi_threshold) & (df["close"] <= df["bb_lower"])
    signals = df[condition].copy()

    if signals.empty:
        return pd.DataFrame()

    hold_minutes = INTERVAL_MINUTES[hold_interval]
    records = []
    cooldown_until = None

    for idx, row in signals.iterrows():
        if cooldown_until and row["close_time"] < cooldown_until:
            continue

        entry_price = row["close"]
        target_time = row["close_time"] + pd.Timedelta(minutes=hold_minutes)
        cooldown_until = target_time
        stop_price = entry_price * (1 - STOP_LOSS_PCT) if use_stop else None

        exit_price = None
        exit_time = None
        exit_reason = "Pending"

        for forward_idx in range(idx + 1, len(df)):
            future_row = df.iloc[forward_idx]

            if use_stop and stop_price is not None and future_row["low"] <= stop_price:
                exit_price = stop_price
                exit_time = future_row["open_time"]
                exit_reason = "Stop"
                break

            if future_row["close_time"] >= target_time:
                exit_price = float(future_row["close"])
                exit_time = future_row["close_time"]
                exit_reason = "Timer"
                break

        pnl_pct = None
        outcome = "Pending"
        if exit_price is not None:
            pnl_pct = (exit_price - row["close"]) / row["close"] * 100
            if pnl_pct > 0:
                outcome = "Gain"
            elif pnl_pct < 0:
                outcome = "Loss"
            else:
                outcome = "Flat"

        records.append(
            {
                "Symbol": symbol_name,
                "Interval": hold_interval,
                "Signal Time": format_utc(row["close_time"]),
                "Entry Price": round(row["close"], 6),
                "RSI": round(row["rsi"], 2),
                "BB Lower": round(row["bb_lower"], 6),
                "Exit Time": format_utc(exit_time),
                "Exit Price": round(exit_price, 6) if exit_price else None,
                "Hold (min)": hold_minutes,
                "PnL %": round(pnl_pct, 2) if pnl_pct is not None else None,
                "Outcome": outcome,
                "Exit Reason": exit_reason,
            }
        )

    return pd.DataFrame(records)


if run_scan:
    if not symbol:
        st.warning("Please provide a symbol (e.g., XRP, BTC, ETH).")
    else:
        if lookback_mode == "Custom dates" and not range_valid:
            st.error("Please choose a valid date range where the end is after the start.")
        else:
            with st.spinner("Fetching historical data from MEXC..."):
                history = load_history(
                    symbol,
                    days if lookback_mode == "Relative days" else None,
                    start_iso if lookback_mode == "Custom dates" else None,
                    end_iso if lookback_mode == "Custom dates" else None,
                )

        if history.empty:
            st.error("Could not load any candles. Check the symbol or try a shorter lookback.")
        else:
            signals_df = build_signals(history, symbol, interval, use_stop_loss)

            if signals_df.empty:
                st.info("No signals found for the selected configuration.")
            else:
                # Summary metrics
                valid_pnls = signals_df["PnL %"].dropna()
                wins = (valid_pnls > 0).sum()
                losses = (valid_pnls < 0).sum()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Signals", len(signals_df))
                win_rate = wins / len(valid_pnls) * 100 if len(valid_pnls) else 0
                col2.metric("Win rate", f"{win_rate:.1f}%")
                avg_pnl = valid_pnls.mean() if len(valid_pnls) else 0
                col3.metric("Avg PnL", f"{avg_pnl:.2f}%")
                total_pnl = valid_pnls.sum() if len(valid_pnls) else 0
                col4.metric("Total PnL", f"{total_pnl:.2f}%")

                def pnl_style(value):
                    if pd.isna(value):
                        return ""
                    return "color:#1a7f37;font-weight:bold;" if value > 0 else "color:#d22d2d;font-weight:bold;"

                styled = (
                    signals_df.style.format(
                        {
                            "Entry Price": "{:.6f}",
                            "BB Lower": "{:.6f}",
                            "Exit Price": "{:.6f}",
                            "PnL %": "{:.2f}",
                        },
                        na_rep="—",
                    )
                    .applymap(pnl_style, subset=["PnL %"])
                )

                st.dataframe(styled, use_container_width=True, height=480)

                st.caption(
                    "PnL simulates buying at the signal close and selling after the configured hold duration."
                    if not use_stop_loss else
                    "PnL simulates buying at the signal close, selling after the hold duration, or earlier if the -1% stop is hit."
                )
else:
    st.info("Configure your scan in the sidebar and click *Find Signals* to begin.")
