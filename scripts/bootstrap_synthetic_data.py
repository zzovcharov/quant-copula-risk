"""
bootstrap_synthetic_data.py -- generate synthetic ETF price data
and save as the cached parquet file expected by quantcopula.data.get_prices().

Run once before the pipeline if you cannot fetch real data via yfinance.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration (match your pipeline parameters)
# ---------------------------------------------------------
tickers = ["XLK", "XLF", "XLI", "XLV"]
start = "2013-01-01"
end = "2023-12-31"


# ---------------------------------------------------------
# Generate synthetic daily prices using geometric Brownian motion
# ---------------------------------------------------------
print("📈 Generating synthetic price data ...")

dates = pd.date_range(start=start, end=end, freq="B")
n_days = len(dates)
n_assets = len(tickers)

np.random.seed(42)

# annual expected returns & volatilities per "ETF"
mus = np.array([0.10, 0.07, 0.06, 0.08])
sigmas = np.array([0.20, 0.18, 0.16, 0.17])

dt = 1.0 / 252

prices_dict = {}

for i, t in enumerate(tickers):
    mu = mus[i]
    sigma = sigmas[i]

    # log-return simulation
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    shocks = np.random.normal(loc=drift, scale=vol, size=n_days)

    log_prices = np.cumsum(shocks)
    start_price = 100.0 + 10 * i   # different starting prices
    prices = start_price * np.exp(log_prices)

    prices_dict[t] = prices

prices_df = pd.DataFrame(prices_dict, index=dates)
prices_df.index.name = "Date"

print("Synthetic prices shape:", prices_df.shape)
print(prices_df.head())


# ---------------------------------------------------------
# Save to the exact cache path expected by data.py
# ---------------------------------------------------------
cache_dir = os.path.join("data", "raw")
os.makedirs(cache_dir, exist_ok=True)

cache_path = os.path.join(
    cache_dir,
    "prices_XLK,XLF,XLI,XLV_2013-01-01_2023-12-31_close.parquet"
)

prices_df.to_parquet(cache_path)
print(f"✅ Synthetic price data saved to:\n   {cache_path}")
print("You can now run: python scripts/run_pipeline.py")
