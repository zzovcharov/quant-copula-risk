# Copula-Based Portfolio Risk Modeling
_A full pipeline for tail-risk analysis using Student-t marginals & Gaussian copula simulation_

This project implements an end-to-end quantitative risk modeling workflow for a **sector ETF portfolio** (`XLK`, `XLF`, `XLI`, `XLV`).  
It estimates **Student-t marginal distributions**, fits a **Gaussian copula** to capture dependence, and computes **portfolio VaR & CVaR** using simulated returns.

The pipeline is designed to run **even without internet access** through synthetic price data generation.

---

## Features

| Component | Description |
|----------|------------|
| **Synthetic or real data support** | Download from Yahoo Finance **or** bootstrap locally |
| **Student-t marginals** | Models fat-tailed return distributions |
| **Gaussian copula modeling** | Captures cross-asset dependence structure |
| **Monte-Carlo simulation** | Generates synthetic portfolio returns with realistic dependencies |
| **Risk metrics** | Computes left-tail **VaR** and **CVaR** |
| **Visualizations** | ECDF, histogram, heatmap, price & returns plots |
| **Modular package** | Extendable architecture in `quantcopula/` |

---

## Project Structure

quant-copula-risk/
├─ data/
│ ├─ raw/ # cached or synthetic prices (parquet)
│ └─ processed/ # t-params, copula correlation, risk summary
│
├─ notebooks/
│ ├─ data/
│ └─ figures/
│
├─ quantcopula/
│ ├─ data.py
│ ├─ margins.py
│ ├─ copula.py
│ ├─ plotting.py
│ └─ risk.py
│
└─ scripts/
├─ bootstrap_synthetic_data.py # generate synthetic prices (offline mode)
└─ run_pipeline.py # main workflow


---

## Installation

```bash
git clone <your-github-url>.git
cd quant-copula-risk

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Pipeline
-  Option A — Real market data (if internet works)
python scripts/run_pipeline.py

##  Option B — Offline mode (recommended)
python scripts/bootstrap_synthetic_data.py
python scripts/run_pipeline.py

This creates the synthetic dataset:
data/raw/prices_XLK,XLF,XLI,XLV_2013-01-01_2023-12-31_close.parquet

## Outputs
| Location          | Contents                                                               |
| ----------------- | ---------------------------------------------------------------------- |
| `data/processed/` | `t_params.parquet`, `gaussian_copula_corr.parquet`, `risk_summary.csv` |
| `figures/`        | Visualizations: prices, returns, heatmap, VaR/CVaR                     |

Example console output:
VaR 95%: -0.0088, CVaR 95%: -0.0110

=== Risk comparison (portfolio level) ===
     model  alpha       VaR      CVaR
historical   0.95 -0.008909 -0.011180
    copula   0.95 -0.008784 -0.010951

## Methodology Overview

1. Data sourcing

- Load from Yahoo or synthetic

- Cache locally as parquet

2. Marginal modeling

- Convert prices → log returns

- Fit Student-t distribution per asset

- Probability Integral Transform (PIT) → returns → uniforms

3. Dependence modeling

- Transform uniforms → Gaussian space

- Estimate empirical correlation matrix

- Fit **Gaussian copula**

- Simulate correlated samples

4. Inverse transformation

- Copula uniforms → simulated returns using t-inverse CDF

5. Portfolio risk

- Equal-weight portfolio (default)

- Compute VaR and CVaR

## Example Plots

Generated in figures/:

- prices.png

- returns.png

- correlation_heatmap.png

- portfolio_distribution.png

- ecdf_hist.png

- ecdf_copula.png

## Troubleshooting

| Problem                | Solution                                        |
| ---------------------- | ----------------------------------------------- |
| Yahoo download fails   | Run offline mode: `bootstrap_synthetic_data.py` |
| Missing parquet engine | `pip install pyarrow`                           |
| No figures saved       | Ensure pipeline writes to existing `figures/`   |
| VaR/CVaR = nan         | Check returns length > 1                        |

## License

MIT License — free for academic and commercial use.

## Author
**Zdravko Ovcharov**
MSc Business Analytics / Data Science