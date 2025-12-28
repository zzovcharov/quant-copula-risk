"""
quantcopula — Copula-based portfolio risk modeling

Submodules
---------
- data: price download / loading utilities
- margins: Student-t marginals + PIT
- copula: Gaussian copula fitting & simulation
- plotting: visualization helpers
- risk: portfolio VaR / CVaR and model comparison
"""

from .data import download_prices, get_prices
from .margins import (
    compute_log_returns,
    fit_t_marginals,
    fit_student_t_params,
    to_uniforms,
    from_uniforms,
)
from .copula import fit_gaussian_copula, simulate_copula, inverse_transform
from .risk import compute_portfolio_var_cvar, compare_models

__all__ = [
    "download_prices",
    "get_prices",
    "compute_log_returns",
    "fit_t_marginals",
    "fit_student_t_params",
    "to_uniforms",
    "from_uniforms",
    "fit_gaussian_copula",
    "simulate_copula",
    "inverse_transform",
    "compute_portfolio_var_cvar",
    "compare_models",
]
