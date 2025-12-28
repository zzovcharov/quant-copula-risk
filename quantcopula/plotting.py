"""
plotting.py — Visualization utilities for copula-based risk modeling
====================================================================

All plots are saved to disk (PNG format) inside `figures/`.
Figures are closed after saving to avoid memory growth in notebooks.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")

__all__ = [
    "plot_prices",
    "plot_returns",
    "plot_heatmap",
    "plot_correlation_heatmap",
    "plot_hist",
    "plot_ecdf",
    "plot_portfolio_distribution",
]


# ----------------------------- helpers -----------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directory for file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _default_path(filename: str) -> str:
    """Send all figures to figures/ by default."""
    return os.path.join("figures", filename)


def _to_portfolio_series(returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Ensure we have a 1D portfolio return series.

    - If a Series is passed, return it as-is.
    - If a DataFrame is passed, assume equal-weight portfolio over columns.
    """
    if isinstance(returns, pd.Series):
        return returns

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas Series or DataFrame.")

    if returns.shape[1] == 0:
        raise ValueError("returns DataFrame has no columns.")

    # Equal-weight portfolio over all assets
    n_assets = returns.shape[1]
    weights = np.full(n_assets, 1.0 / n_assets)
    port = returns.values @ weights
    return pd.Series(port, index=returns.index, name="portfolio")


# ----------------------------- core plots -----------------------------

def plot_prices(px: pd.DataFrame, out_path: Optional[str] = None) -> None:
    if out_path is None:
        out_path = _default_path("prices.png")
    _ensure_dir(out_path)

    plt.figure(figsize=(12, 6))
    px.plot(lw=1.5)
    plt.title("Sector ETF Price Levels")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_returns(log_returns: pd.DataFrame, out_path: Optional[str] = None) -> None:
    if out_path is None:
        out_path = _default_path("returns.png")
    _ensure_dir(out_path)

    plt.figure(figsize=(12, 6))
    log_returns.plot(lw=1.0)
    plt.title("Daily Log Returns")
    plt.ylabel("Log Return")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(log_returns: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Actual implementation of the correlation heatmap.
    """
    if out_path is None:
        out_path = _default_path("correlation_heatmap.png")
    _ensure_dir(out_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(log_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Log Returns")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap(log_returns: pd.DataFrame, out_path: Optional[str] = None) -> None:
    """
    Thin wrapper to match run_pipeline.py import.
    """
    return plot_correlation_heatmap(log_returns, out_path)


def plot_hist(series: pd.Series, title: str = "Distribution", out_path: Optional[str] = None) -> None:
    if out_path is None:
        out_path = _default_path("hist.png")
    _ensure_dir(out_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(series, bins=80, kde=True)
    plt.title(title)
    plt.xlabel("Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------- ECDF plot -----------------------------

def plot_ecdf(series: pd.Series, lines: Dict[str, float], out_path: Optional[str] = None) -> None:
    """
    Compute and plot ECDF of portfolio returns with vertical VaR/CVaR lines.
    """
    if out_path is None:
        out_path = _default_path("ecdf_var_cvar.png")
    _ensure_dir(out_path)

    x = np.sort(series.values)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Empirical CDF")

    for label, value in lines.items():
        plt.axvline(value, linestyle="--", label=f"{label}: {value:.4f}")

    plt.title("Portfolio Return ECDF with Risk Thresholds")
    plt.xlabel("Portfolio Return")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------- portfolio dist -----------------------------

def plot_portfolio_distribution(
        returns: Union[pd.Series, pd.DataFrame],
        alpha: float = 0.95,
        out_path: Optional[str] = None,
) -> None:
    """
    Plot histogram of portfolio returns with VaR and CVaR markers.

    Parameters
    ----------
    returns : Series or DataFrame
        - If Series: interpreted directly as portfolio returns.
        - If DataFrame: an equal-weight portfolio over columns is used.
    alpha : float
        Confidence level for VaR / CVaR (e.g. 0.95 for 95%).
    out_path : str, optional
        Where to save the figure (PNG). Defaults to figures/portfolio_distribution.png.
    """
    if out_path is None:
        out_path = _default_path("portfolio_distribution.png")
    _ensure_dir(out_path)

    series = _to_portfolio_series(returns)

    # Left-tail risk: lower (1 - alpha) quantile
    q = np.percentile(series.values, (1 - alpha) * 100)
    cvar = series[series <= q].mean()

    plt.figure(figsize=(10, 6))
    sns.histplot(series, bins=80, kde=True)
    plt.axvline(q, color="orange", linestyle="--", label=f"VaR {int(alpha*100)}%: {q:.4f}")
    plt.axvline(cvar, color="red", linestyle="--", label=f"CVaR {int(alpha*100)}%: {cvar:.4f}")
    plt.title(f"Portfolio Return Distribution (Confidence {alpha:.2f})")
    plt.xlabel("Portfolio Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
