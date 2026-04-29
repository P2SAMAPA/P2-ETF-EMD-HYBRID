# P2-ETF-EMD-HYBRID

Empirical Mode Decomposition (EMD) + Hybrid Forecasting for ETF absolute returns.

## Overview

This engine decomposes daily ETF returns into **Intrinsic Mode Functions (IMFs)** using CEEMDAN (adaptive, data‑driven decomposition). For each IMF, it trains three models:
- Support Vector Regression (SVR)
- Multi‑Layer Perceptron (MLP)
- LightGBM

The best model (lowest validation MSE) is selected per IMF. Forecasts from all IMFs are summed to produce the next day’s **absolute return** prediction. The engine runs two modes:

- **Global**: Train on 80% of all available data (2008–present), validate on next 10%, then forecast the next trading day.
- **Shrinking windows**: 3‑year sliding windows (2008–2010, 2009–2011, … up to 2022–2024). For each window, the same hybrid procedure produces a top ETF; consensus across windows yields a final pick with conviction score.

All results are pushed daily to a Hugging Face dataset.

## Repository Structure
.
├── .github/workflows/daily_run.yml # GitHub Actions cron job
├── config.py # All parameters and universe definitions
├── data_manager.py # Load master data, align macro, prepare returns
├── emd_hybrid_model.py # EMD decomposition + per‑IMF hybrid forecaster
├── trainer.py # Main script (global + shrinking windows)
├── push_results.py # Upload results to HF dataset
├── us_calendar.py # US trading calendar utilities
└── README.md

text

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/P2SAMAPA/P2-ETF-EMD-HYBRID.git
   cd P2-ETF-EMD-HYBRID
