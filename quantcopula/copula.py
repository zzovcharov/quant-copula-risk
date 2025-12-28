"""
copula.py — Gaussian copula fitting and simulation
==================================================

This module works with PIT-transformed uniform margins (U in [0,1])
to fit a Gaussian copula and simulate new dependent samples.

Public API
----------
- fit_gaussian_copula(U) -> dict with 'corr' and 'columns'
- simulate_copula(copula, n_samples, random_state=None) -> pd.DataFrame of U
- inverse_transform(simulated_u, reference_returns, params_path=None) -> returns

The inverse_transform uses the Student-t marginals previously fitted and
saved by margins.fit_t_marginals() to map uniforms back to returns.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import os

import numpy as np
import pandas as pd
from scipy import stats

from quantcopula.margins import frame_to_params, from_uniforms

__all__ = [
    "fit_gaussian_copula",
    "simulate_copula",
    "inverse_transform",
]

# Default path for the t-parameters saved by margins.fit_t_marginals
_DEFAULT_T_PARAMS_PATH = os.path.join("data", "processed", "t_params.parquet")


# ---------------------------------------------------------------------
# Gaussian copula representation
# ---------------------------------------------------------------------

def _to_gaussian_space(U: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Map uniforms U in [0,1] to standard Normal space via inverse CDF (ppf).

    We clip U away from 0 and 1 to avoid infinities.
    """
    if not isinstance(U, pd.DataFrame):
        raise TypeError("U must be a pandas DataFrame of uniforms in [0,1].")

    Uc = U.astype(float).clip(eps, 1 - eps)
    Z = stats.norm.ppf(Uc)
    Z = pd.DataFrame(Z, index=U.index, columns=U.columns)
    return Z


def fit_gaussian_copula(U: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a Gaussian copula to PIT-transformed uniform data U.

    Parameters
    ----------
    U : pd.DataFrame
        Uniform(0,1) margins, columns = assets, rows = observations.

    Returns
    -------
    copula : dict
        {
            "corr": np.ndarray (d x d correlation matrix),
            "columns": list of column names (asset order),
        }
    """
    if not isinstance(U, pd.DataFrame):
        raise TypeError("U must be a pandas DataFrame.")

    if U.isna().any().any():
        raise ValueError("U contains NaNs; please clean data before fitting copula.")

    Z = _to_gaussian_space(U)  # map to standard normals

    # Estimate correlation matrix in Gaussian space
    # np.corrcoef with rowvar=False → columns = variables
    corr = np.corrcoef(Z.values, rowvar=False)

    if corr.shape[0] != corr.shape[1]:
        raise RuntimeError("Correlation matrix is not square; something went wrong in fitting.")

    # Ensure symmetry and positive-definiteness (basic symmetrization)
    corr = (corr + corr.T) / 2.0

    copula = {
        "corr": corr,
        "columns": list(U.columns),
    }

    # Optionally, save correlation matrix for inspection
    os.makedirs("data/processed", exist_ok=True)
    corr_df = pd.DataFrame(corr, index=U.columns, columns=U.columns)
    corr_df.to_parquet(os.path.join("data", "processed", "gaussian_copula_corr.parquet"))
    print("✅ Fitted Gaussian copula and saved correlation matrix to data/processed/")

    return copula


# ---------------------------------------------------------------------
# Simulation from Gaussian copula
# ---------------------------------------------------------------------

def simulate_copula(
        copula: Dict[str, Any],
        n_samples: int,
        random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate Uniform(0,1) samples from a fitted Gaussian copula.

    Parameters
    ----------
    copula : dict
        Output of fit_gaussian_copula(), must contain "corr" and "columns".
    n_samples : int
        Number of simulated scenarios.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    U_sim : pd.DataFrame
        Simulated uniforms in [0,1], shape (n_samples, d), columns = assets.
    """
    corr = np.asarray(copula.get("corr"))
    cols = list(copula.get("columns", []))

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("copula['corr'] must be a square 2D array.")
    d = corr.shape[0]

    if len(cols) != d:
        raise ValueError("Length of copula['columns'] must match corr dimension.")

    rng = np.random.default_rng(random_state)

    # Cholesky decomposition for correlation structure
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fall back: add a small jitter to the diagonal if not PD
        eps = 1e-8
        jitter = eps * np.eye(d)
        L = np.linalg.cholesky(corr + jitter)
        print("⚠️ Correlation matrix not PD; added small jitter to diagonal.")

    # Generate independent standard normals
    Z = rng.standard_normal(size=(n_samples, d))
    # Impose correlation: X = Z * L^T
    X = Z @ L.T

    # Map back to uniforms via Normal CDF
    U_sim = stats.norm.cdf(X)
    U_sim = pd.DataFrame(U_sim, columns=cols)

    return U_sim


# ---------------------------------------------------------------------
# Inverse transform: uniforms -> returns using t-marginals
# ---------------------------------------------------------------------

def inverse_transform(
        simulated_u: pd.DataFrame,
        reference_returns: pd.DataFrame,
        params_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map simulated uniforms back to return space using fitted Student-t marginals.

    This uses the t-parameters saved by margins.fit_t_marginals(), which stores
    them by default at data/processed/t_params.parquet.

    Parameters
    ----------
    simulated_u : pd.DataFrame
        Simulated Uniform(0,1) data, columns = assets.
    reference_returns : pd.DataFrame
        Historical returns used for fitting; only its columns (asset order)
        are used here to align the dimensions.
    params_path : str, optional
        Path to the saved t-parameters parquet file. If None, uses the
        default data/processed/t_params.parquet.

    Returns
    -------
    sim_returns : pd.DataFrame
        Simulated returns with the same columns (assets) as reference_returns.
    """
    if not isinstance(simulated_u, pd.DataFrame):
        raise TypeError("simulated_u must be a pandas DataFrame.")
    if not isinstance(reference_returns, pd.DataFrame):
        raise TypeError("reference_returns must be a pandas DataFrame.")

    if params_path is None:
        params_path = _DEFAULT_T_PARAMS_PATH

    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"t-parameters file not found at {params_path}. "
            "Make sure fit_t_marginals() has been called before inverse_transform()."
        )

    # Load fitted t-parameters
    params_df = pd.read_parquet(params_path)
    params = frame_to_params(params_df)

    # Align simulated uniforms to the reference asset order
    cols = list(reference_returns.columns)
    missing = set(cols) - set(simulated_u.columns)
    if missing:
        raise ValueError(f"simulated_u is missing columns: {sorted(missing)}")

    U_aligned = simulated_u[cols].copy()
    U_aligned.index = range(len(U_aligned))  # simple integer index

    # Use margins.from_uniforms to map back to returns
    sim_returns = from_uniforms(U_aligned, params)

    return sim_returns
