"""
data.py — Data download and preprocessing utilities
"""

from __future__ import annotations

import os
from typing import Iterable
import pandas as pd
import yfinance as yf


def _slug(s: str) -> str:
    """Create a filesystem-safe slug from a string."""
    return "".join(ch if ch.isalnum() or ch in "-_," else "_" for ch in s)


def _default_cache_path(
        tickers: Iterable[str],
        start: str,
        end: str,
        auto_adjust: bool,
        cache_dir: str = "data/raw",
) -> str:
    """
    Build a deterministic cache filename for a given ticker/date combo.
    """
    tickers = list(tickers)
    tks = ",".join(tickers)
    fname = f"prices_{_slug(tks)}_{start}_{end}_{'adj' if auto_adjust else 'close'}.parquet"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, fname)


def get_prices(
        tickers: Iterable[str],
        start: str,
        end: str,
        cache_path: str | None = None,
        auto_adjust: bool = False,
        interval: str = "1d",
) -> pd.DataFrame:
    """
    Robust per-ticker download to avoid yfinance multi-ticker failures.
    Falls back to cache if available.

    Parameters
    ----------
    tickers : iterable of str
        List of ticker symbols, e.g. ['XLK', 'XLF'].
    start, end : str
        Date bounds in 'YYYY-MM-DD' format.
    cache_path : str, optional
        Explicit cache path. If None, a unique file name is derived
        under data/raw based on tickers and dates.
    auto_adjust : bool
        yfinance auto_adjust flag.
    interval : str
        Data frequency, e.g. '1d', '1wk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with DateTime index and one column per ticker (floats).
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")

    # Build unique cache file if not provided
    if cache_path is None:
        cache_path = _default_cache_path(tickers, start, end, auto_adjust)

    # ------------------- cache handling -------------------
    if os.path.exists(cache_path):
        print(f"📂 Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"⬇ Downloading tickers one-by-one: {tickers}")

    frames = []
    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                interval=interval,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"⚠️ Failed to download {t}: {e}")
            continue

        if df is None or len(df) == 0:
            print(f"⚠️ No data for {t}")
            continue

        # choose Adj Close or Close
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df = df.sort_index()
        series = df[col].astype("float64")
        frames.append(series.rename(t))

    if not frames:
        raise RuntimeError(
            f"❌ Could not download any data for {tickers}. "
            "This is likely a network issue or a temporary Yahoo Finance outage.\n"
            f"Expected cache path (if you want to provide data manually): {cache_path}"
        )

    # Combine into final price table
    prices = pd.concat(frames, axis=1).dropna(how="any").sort_index()
    prices.index = pd.to_datetime(prices.index)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"Saved downloaded prices to {cache_path}")

    return prices


def download_prices(
        tickers: Iterable[str],
        start: str,
        end: str,
        auto_adjust: bool = False,
        interval: str = "1d",
        cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper used by the pipeline.

    This matches the expected interface in run_pipeline.py.

    Parameters
    ----------
    tickers : iterable of str
        List of ticker symbols.
    start, end : str
        Date bounds in 'YYYY-MM-DD' format.
    auto_adjust : bool, optional
        Whether to auto-adjust prices via yfinance.
    interval : str, optional
        Data frequency, e.g. '1d'.
    cache_path : str, optional
        Custom cache path; if None, uses a default under data/raw.

    Returns
    -------
    pd.DataFrame
        DataFrame of prices with DateTime index and one column per ticker.
    """
    return get_prices(
        tickers=tickers,
        start=start,
        end=end,
        cache_path=cache_path,
        auto_adjust=auto_adjust,
        interval=interval,
    )
