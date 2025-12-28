"""
risk.py — Portfolio risk metrics and model comparison
=====================================================

This module provides:
- compute_portfolio_var_cvar: compute portfolio VaR and CVaR
- compare_models: compare historical vs copula-simulated risk
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from quantcopula.plotting import plot_ecdf  # optional visuals

__all__ = ["compute_portfolio_var_cvar", "compare_models"]

RESULTS_DIR = "data/processed"
RISK_SUMMARY_FILE = "risk_summary.csv"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_portfolio_series(
        returns: Union[pd.Series, pd.DataFrame],
        weights: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Ensure we have a 1D portfolio return series.

    - If a Series is passed, return it as-is.
    - If a DataFrame is passed, build a (possibly equal-weight) portfolio.
    """
    if isinstance(returns, pd.Series):
        return returns

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas Series or DataFrame.")

    if returns.shape[1] == 0:
        raise ValueError("returns DataFrame has no columns.")

    n_assets = returns.shape[1]
    if weights is None:
        weights = np.full(n_assets, 1.0 / n_assets)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != n_assets:
            raise ValueError(
                f"weights length {weights.shape[0]} does not match number of assets {n_assets}"
            )
        # normalize just in case
        s = weights.sum()
        if s == 0:
            raise ValueError("weights sum to zero.")
        weights = weights / s

    port = returns.values @ weights
    return pd.Series(port, index=returns.index, name="portfolio")


def compute_portfolio_var_cvar(
        returns: Union[pd.Series, pd.DataFrame],
        alpha: float = 0.95,
        weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute left-tail portfolio VaR and CVaR at confidence level alpha.

    Parameters
    ----------
    returns : Series or DataFrame
        If Series: portfolio returns.
        If DataFrame: asset returns; an equal-weight (or weighted) portfolio is built.
    alpha : float
        Confidence level, e.g. 0.95 for 95% VaR/CVaR.
    weights : np.ndarray, optional
        Portfolio weights (length = number of columns in returns).

    Returns
    -------
    VaR, CVaR : float, float
        Left-tail VaR and CVaR at level alpha (typically negative numbers).
    """
    series = _to_portfolio_series(returns, weights)

    # Left-tail risk: lower (1 - alpha) quantile
    q = np.percentile(series.values, (1 - alpha) * 100)
    cvar = series[series <= q].mean()

    return float(q), float(cvar)


def compare_models(
        hist_returns: Union[pd.Series, pd.DataFrame],
        sim_returns: Union[pd.Series, pd.DataFrame],
        alpha: float = 0.95,
        weights: Optional[np.ndarray] = None,
        save: bool = True,
) -> pd.DataFrame:
    """
    Compare historical vs copula-simulated portfolio risk.

    Parameters
    ----------
    hist_returns : Series or DataFrame
        Historical asset/portfolio returns.
    sim_returns : Series or DataFrame
        Simulated asset/portfolio returns from the copula model.
    alpha : float
        Confidence level for VaR/CVaR.
    weights : np.ndarray, optional
        Portfolio weights if returns are DataFrames.
    save : bool
        If True, save a CSV summary and ECDF plots.

    Returns
    -------
    summary : pd.DataFrame
        Table with VaR/CVaR for historical vs copula models.
    """
    hist_port = _to_portfolio_series(hist_returns, weights)
    sim_port = _to_portfolio_series(sim_returns, weights)

    hist_VaR, hist_CVaR = compute_portfolio_var_cvar(hist_port, alpha)
    sim_VaR, sim_CVaR = compute_portfolio_var_cvar(sim_port, alpha)

    summary = pd.DataFrame(
        {
            "model": ["historical", "copula"],
            "alpha": [alpha, alpha],
            "VaR": [hist_VaR, sim_VaR],
            "CVaR": [hist_CVaR, sim_CVaR],
        }
    )

    if save:
        _ensure_dir(RESULTS_DIR)
        out_path = os.path.join(RESULTS_DIR, RISK_SUMMARY_FILE)
        summary.to_csv(out_path, index=False)
        print(f"✅ Saved risk summary to {out_path}")

        # ECDF plots for both models
        _ensure_dir("figures")
        plot_ecdf(
            hist_port,
            {
                f"Hist VaR {int(alpha*100)}%": hist_VaR,
                f"Hist CVaR {int(alpha*100)}%": hist_CVaR,
            },
            out_path=os.path.join("figures", "ecdf_hist.png"),
        )
        plot_ecDF_lines = {
            f"Copula VaR {int(alpha*100)}%": sim_VaR,
            f"Copula CVaR {int(alpha*100)}%": sim_CVaR,
        }
        plot_ecdf(
            sim_port,
            plot_ecDF_lines,
            out_path=os.path.join("figures", "ecdf_copula.png"),
        )

    print("\n=== Risk comparison (portfolio level) ===")
    print(summary.to_string(index=False))

    return summary
