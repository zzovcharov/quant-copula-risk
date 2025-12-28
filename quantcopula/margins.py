"""
margins.py — Student-t marginals and Probability Integral Transforms (PIT)
===========================================================================

What this module provides
-------------------------
1) Fit a univariate Student-t distribution to each return series (per column).
2) Transform returns → uniforms in [0,1] via PIT (using the fitted t CDF).
3) Transform uniforms → returns via inverse PIT (using the fitted t PPF).

Public API
----------
- compute_log_returns(prices) -> pd.DataFrame
- fit_student_t_params(returns) -> dict[col] = (df, loc, scale)
- fit_t_marginals(returns, save_path=None) -> uniforms DataFrame   <-- added
- to_uniforms(returns, params) -> DataFrame of U in [0,1]
- from_uniforms(U, params, eps=1e-12) -> DataFrame of simulated returns
- params_to_frame(params) <-> frame_to_params(df)
- ks_uniform_test(U) -> pd.DataFrame (optional diagnostic)
"""

from __future__ import annotations

from typing import Dict, Tuple, Mapping
import os  # <-- added

import numpy as np
import pandas as pd
from scipy import stats

# Type alias for clarity: each column maps to (df, loc, scale)
Params = Dict[str, Tuple[float, float, float]]

__all__ = [
    "compute_log_returns",
    "fit_student_t_params",
    "fit_t_marginals",    # <-- added
    "to_uniforms",
    "from_uniforms",
    "params_to_frame",
    "frame_to_params",
    "ks_uniform_test",
]

# Default path for saving fitted t-parameters
_PROCESSED_DIR = "data/processed"
_T_PARAMS_FILE = "t_params.parquet"


# ---------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price levels.

    Returns
    -------
    pd.DataFrame
        Log returns aligned to prices.index with the first row dropped.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    lr = np.log(prices / prices.shift(1))
    return lr.dropna(how="any")


# ---------------------------------------------------------------------
# Fitting Student-t marginals
# ---------------------------------------------------------------------

def _fit_t_safe(x: pd.Series) -> Tuple[float, float, float]:
    """
    Fit a Student-t distribution to a 1D series using MLE.
    If the series is (near) constant, return a safe fallback.

    Returns
    -------
    (df, loc, scale)
    """
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        # Degenerate: no data. Use minimal parameters.
        return (5.0, 0.0, 1e-8)

    std = float(x.std(ddof=0))
    if std <= 1e-12:
        # Near-constant: t-fit unstable. Use tiny variance normal proxy via t with large df.
        return (1e6, float(x.mean()), 1e-8)

    # Regular MLE fit
    df, loc, scale = stats.t.fit(x.values)
    # Guard against pathological scale
    scale = float(max(scale, 1e-12))
    return float(df), float(loc), scale


def fit_student_t_params(returns: pd.DataFrame) -> Params:
    """
    Fit a univariate Student-t to each column of returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Columns = assets, rows = timestamps, values = returns.

    Returns
    -------
    params : dict[str, tuple[float, float, float]]
        Mapping {column_name: (df, loc, scale)}.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    params: Params = {}
    for col in returns.columns:
        params[col] = _fit_t_safe(returns[col])
    return params


# NEW: high-level helper used by run_pipeline.py
def fit_t_marginals(returns: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    """
    Fit Student-t marginals and return uniform PIT-transformed data.

    This is the function expected by run_pipeline.py:
        uniform_margins = fit_t_marginals(log_returns)

    It also saves the fitted parameters to data/processed/t_params.parquet
    by default, so they can be reused later (e.g. in inverse transforms).

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns.
    save_path : str, optional
        Where to save the fitted parameters. If None, uses the default
        under data/processed.

    Returns
    -------
    U : pd.DataFrame
        PIT-transformed uniforms in [0,1], same shape as `returns`.
    """
    if save_path is None:
        os.makedirs(_PROCESSED_DIR, exist_ok=True)
        save_path = os.path.join(_PROCESSED_DIR, _T_PARAMS_FILE)

    params = fit_student_t_params(returns)
    params_df = params_to_frame(params)
    params_df.to_parquet(save_path, index=False)
    print(f"✅ Saved t-parameters to {save_path}")

    U = to_uniforms(returns, params)
    return U


# ---------------------------------------------------------------------
# PIT: returns <-> uniforms
# ---------------------------------------------------------------------

def _ensure_params_cover(columns: pd.Index, params: Mapping[str, Tuple[float, float, float]]) -> None:
    missing = set(columns) - set(params.keys())
    if missing:
        raise ValueError(f"Missing params for columns: {sorted(missing)}")


def to_uniforms(returns: pd.DataFrame, params: Params) -> pd.DataFrame:
    """
    Apply Probability Integral Transform (PIT) per column:
    U = F_t(r | df, loc, scale)  -> U ~ Uniform(0,1) under the fitted t.

    Parameters
    ----------
    returns : pd.DataFrame
    params  : dict mapping col -> (df, loc, scale)

    Returns
    -------
    U : pd.DataFrame of same shape/index/columns as `returns`
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    _ensure_params_cover(returns.columns, params)

    U = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        df, loc, scale = params[col]
        U[col] = stats.t.cdf(returns[col].astype(float), df, loc=loc, scale=scale)
    return U


def from_uniforms(U: pd.DataFrame, params: Params, eps: float = 1e-12) -> pd.DataFrame:
    """
    Inverse PIT per column:
    r = F_t^{-1}(u | df, loc, scale)
    Clips u to (eps, 1-eps) to avoid inf at 0/1.

    Parameters
    ----------
    U     : pd.DataFrame of uniforms in [0,1]
    params: dict mapping col -> (df, loc, scale)
    eps   : float, small positive number to clip uniforms away from 0 and 1

    Returns
    -------
    sim : pd.DataFrame of simulated returns
    """
    if not isinstance(U, pd.DataFrame):
        raise TypeError("U must be a pandas DataFrame.")
    _ensure_params_cover(U.columns, params)

    sim = pd.DataFrame(index=U.index, columns=U.columns, dtype=float)
    # Clip to avoid ±inf from ppf at exactly 0/1
    Uc = U.astype(float).clip(eps, 1 - eps)

    for col in U.columns:
        df, loc, scale = params[col]
        sim[col] = stats.t.ppf(Uc[col], df, loc=loc, scale=scale)
    return sim


# ---------------------------------------------------------------------
# Save / load fitted params
# ---------------------------------------------------------------------

def params_to_frame(params: Params) -> pd.DataFrame:
    """
    Convert params dict to a tidy DataFrame for saving.
    Columns: ['asset', 'df', 'loc', 'scale']
    """
    df = pd.DataFrame.from_dict(params, orient="index", columns=["df", "loc", "scale"])
    df.index.name = "asset"
    return df.reset_index()


def frame_to_params(df: pd.DataFrame) -> Params:
    """
    Convert a DataFrame (as saved by params_to_frame) back to dict form.
    """
    required = {"asset", "df", "loc", "scale"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {set(df.columns)}")
    return {
        str(row["asset"]): (float(row["df"]), float(row["loc"]), float(row["scale"]))
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------
# Optional: simple uniformity diagnostic
# ---------------------------------------------------------------------

def ks_uniform_test(U: pd.DataFrame) -> pd.DataFrame:
    """
    Run a one-sample Kolmogorov–Smirnov test per column against Uniform(0,1).

    Returns
    -------
    pd.DataFrame with index=asset, columns=['KS stat', 'p-value']
    """
    rows = []
    for col in U.columns:
        u = pd.Series(U[col]).dropna().astype(float)
        stat, pval = stats.kstest(u.values, "uniform", args=(0, 1))
        rows.append({"asset": col, "KS stat": stat, "p-value": pval})
    out = pd.DataFrame(rows).set_index("asset")
    return out
