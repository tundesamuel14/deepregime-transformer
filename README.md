DeepRegime: Transformer-Based Market Regime Modeling and Risk-Managed Allocation

This repository contains an end-to-end framework for detecting market regimes using a Transformer-based sequence model and applying those regimes to dynamic risk allocation. The project demonstrates how regime-aware exposure control can improve risk-adjusted returns relative to static buy-and-hold benchmarks.

The workflow includes:

Data preparation and feature engineering

Regime labeling (via KMeans on SPY/VIX structure)

Transformer-based sequence classification

Inference on historical windows

Backtesting dynamic exposure strategies

Evaluation using cumulative return, Sharpe ratio, volatility, and drawdowns

1. Project Overview

Financial markets transition through distinct volatility and trend environments, commonly called "regimes."
Examples include:

Low-volatility, trend-friendly periods

High-volatility, crisis periods

Mean-reversion phases

Risk-off events

Understanding and predicting these regimes is central to professional quantitative investing.
This project implements:

A Transformer encoder that processes the most recent 60 days of SPY/VIX features

A classification head that assigns each window to a regime

A leveraged dynamic exposure strategy driven by the predicted regime

A full backtesting engine to measure economic value

2. Directory Structure
deepregime-transformer/
│
├── deepregime/
│   ├── models/
│   │   └── transformer_regime.py        # Transformer architecture
│   ├── training/
│   │   ├── dataset.py                   # Sliding window dataset construction
│   │   └── train_transformer.py         # Model training loop
│   ├── inference.py                     # Inference for last-N windows
│   └── backtest/
│       └── backtest_leverage.py         # Backtest logic for regime-based strategies
│
├── data/
│   └── regimes/                         # SPY/VIX regime-labelled parquet data
│
├── notebooks/
│   └── Regime_Transformer_Demo.ipynb    # Full demonstration notebook
│
├── plots/
│   ├── regime_predictions.png
│   └── equity_comparison.png
│
└── README.md

3. Model Details
Input Features

The model uses the following daily features over rolling 60-day windows:

1-day, 5-day, 20-day returns

5-day and 20-day realized volatility

VIX index level

Features are standardized using training-sample statistics.

Transformer Architecture

2 Transformer encoder layers

4 attention heads

Model dimension d_model = 64

Feed-forward dimension = 128

Dropout = 0.1

Classification head with cross-entropy loss

Regime Labels

Regimes are precomputed using KMeans clustering on SPY + VIX characteristics:

Regime 0: Low-volatility / stable environments

Regime 1: High-volatility / risk-off environments

4. Regime Prediction

The model generates daily regime predictions and regime probability distributions.

The following figure illustrates predicted versus true regimes:

plots/regime_predictions.png


(Add the image once uploaded.)

5. Backtesting

Three strategies are compared:

1. Buy & Hold SPY

Static long-only exposure.

2. Regime Strategy (Unlevered)

Fully invested only in regime 0

In cash during regime 1

3. Leveraged Regime Strategy

Dynamic exposure based on predicted regime:

1.5× exposure in regime 0

0.3× exposure in regime 1

This reflects practical risk-managed portfolio construction used by professional systematic managers.

6. Performance Summary

Buy & Hold SPY

Cumulative return: 635.50%

Annualized Sharpe: 0.63

Daily mean return: 0.0476%

Daily volatility: 1.2052%

Regime Strategy (Unlevered)

Cumulative return: 205.01%

Annualized Sharpe: 0.66

Daily mean return: 0.0242%

Daily volatility: 0.5824%

Leveraged Regime Strategy

Cumulative return: 589.89%

Annualized Sharpe: 0.74

Daily mean return: 0.0433%

Daily volatility: 0.9290%

7. Interpretation

The leveraged regime strategy achieves:

Higher Sharpe ratio than buy & hold

Substantial reduction in drawdowns

Competitive long-term returns

Significantly smoother equity curve in crisis periods

This demonstrates that:

The Transformer successfully learns meaningful volatility regimes.

Regime predictions can be used for economically valuable risk allocation.

A simple leveraged overlay can outperform buy & hold on a risk-adjusted basis.

This system behaves similarly to practical volatility-aware or risk-parity-style strategies used at quantitative hedge funds.

8. How to Run the Project
Install dependencies
pip install -r requirements.txt

Train the regime Transformer
python -m deepregime.training.train_transformer

Run model inference
python -m deepregime.inference

Backtest dynamic regime strategies
python -m deepregime.backtest.backtest_leverage

9. Future Work

Regime transition prediction (next 5–10 day regime flipping)

Volatility targeting based on realized vol forecasts

Multi-asset regimes (SPY, VIX, TLT, GOLD, credit spreads)

Transaction cost modeling and slippage

Walk-forward train / validation framework

Hyperparameter optimization

Alternative clustering methods (HMMs, Gaussian Mixture Models)

10. License# deepregime-transformer
Transformer-based market regime detection for quant trading.
