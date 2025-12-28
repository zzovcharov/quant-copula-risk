"""
run_pipeline.py

Main orchestration script for the Copula-Based Risk Modeling Project.
It runs all steps:
1. Download and preprocess data
2. Fit t-distribution marginals
3. Fit Gaussian copula and simulate dependencies
4. Compute and visualize risk metrics
"""

# -------------------------------
# Imports from package
# -------------------------------
from quantcopula.data import download_prices
from quantcopula.margins import compute_log_returns, fit_t_marginals
from quantcopula.copula import fit_gaussian_copula, simulate_copula, inverse_transform
from quantcopula.risk import compute_portfolio_var_cvar, compare_models
from quantcopula.plotting import (
    plot_prices,
    plot_returns,
    plot_heatmap,
    plot_portfolio_distribution,
)

# -------------------------------
# Parameters
# -------------------------------
TICKERS = ["XLK", "XLF", "XLI", "XLV"]
START_DATE = "2013-01-01"
END_DATE = "2023-12-31"
N_SAMPLES = 10000
CONFIDENCE_LEVEL = 0.95


def main():
    print("Starting Copula-Based Risk Modeling Pipeline...\n")

    # Download data
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    plot_prices(prices)

    # Compute log returns
    log_returns = compute_log_returns(prices)
    plot_returns(log_returns)
    plot_heatmap(log_returns)

    # Fit t-marginals → uniform margins
    uniform_margins = fit_t_marginals(log_returns)
    print("Uniform marginals computed.\n")

    # Fit Gaussian copula and simulate
    copula = fit_gaussian_copula(uniform_margins)
    simulated_u = simulate_copula(copula, N_SAMPLES)

    # Transform simulated uniforms back to real returns
    simulated_returns = inverse_transform(simulated_u, log_returns)
    print("Simulated returns generated.\n")

    # Compute portfolio risk
    VaR, CVaR = compute_portfolio_var_cvar(simulated_returns, CONFIDENCE_LEVEL)
    print(f"VaR {int(CONFIDENCE_LEVEL*100)}%: {VaR:.4f}, CVaR {int(CONFIDENCE_LEVEL*100)}%: {CVaR:.4f}")

    # Compare with historical model
    compare_models(log_returns, simulated_returns)

    # Visualize portfolio distribution
    plot_portfolio_distribution(simulated_returns, CONFIDENCE_LEVEL)

    print("\n Pipeline complete! Results saved in /data/processed and /figures.")


if __name__ == "__main__":
    main()
