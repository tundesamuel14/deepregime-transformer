DeepRegime: Transformer-Based Market Regime Classification and Risk-Managed Allocation

This repository implements a Transformer-based market regime classifier trained on SPY/VIX features and integrates it into a dynamic exposure strategy. The objective is to demonstrate how sequence models can identify structural market regimes and how these predictions can be used to construct risk-managed trading strategies with improved risk-adjusted performance.

1. Overview

Financial markets exhibit distinct regimes such as high-volatility distress periods and low-volatility expansion periods. Identifying these regimes enables:

Controlled exposure to risk

Downside protection during turbulent periods

Improved Sharpe ratios

More stable long-term performance

This project builds a regime-detection framework using a Transformer classifier and evaluates its economic usefulness in a backtesting environment.

2. Project Structure
deepregime-transformer/
│
├── deepregime/
│   ├── models/
│   │   └── transformer_regime.py       # Transformer architecture
│   ├── training/
│   │   ├── dataset.py                  # Sliding window dataset generation
│   │   └── train_transformer.py        # Training script
│   ├── inference.py                    # Inference utilities
│   └── backtest/
│       └── backtest_leverage.py        # Backtesting logic
│
├── data/
│   └── regimes/                         # Parquet files containing regime labels (SPY + VIX)
│
├── notebooks/
│   └── Regime_Transformer_Demo.ipynb    # End-to-end demonstration notebook
│
├── plots/
│   ├── regime_predictions.png
│   └── equity_comparison.png
│
└── README.md

3. Data and Features

The model uses daily SPY and VIX-derived features:

1-day, 5-day, and 20-day returns

5-day and 20-day realized volatility

VIX index levels

Regime labels are generated using KMeans clustering, which produces two interpretable market states:

Regime 0: Low-volatility, stable market conditions

Regime 1: High-volatility, distressed conditions

The model learns to classify the current market regime using the past 60 days of feature history.

4. Model Architecture

A Transformer encoder is used for sequence modeling:

Sequence length: 60 days

Embedding dimension: 64

Number of layers: 2

Attention heads: 4

Hidden dimension: 128

Dropout: 0.1

Output: Regime classification (0 or 1)

The model is trained using cross-entropy loss with a train/validation split.

5. Backtesting Methodology

Three strategies are evaluated:

Buy & Hold SPY

Baseline benchmark

Regime Filter Strategy

Invest in SPY only during Regime 0

Move to cash during Regime 1

Leveraged Regime Strategy

Apply 1.5× exposure in Regime 0

Apply 0.3× exposure in Regime 1

Represents dynamic, model-driven risk scaling

Equity curves, cumulative returns, daily statistics, and Sharpe ratios are computed.

6. Results
Buy & Hold SPY

Cumulative return: 635.50%

Annualized Sharpe: 0.63

Daily mean return: 0.0476%

Daily volatility: 1.2052%

Regime Filter Strategy

Cumulative return: 205.01%

Annualized Sharpe: 0.66

Daily mean return: 0.0242%

Daily volatility: 0.5824%

Leveraged Regime Strategy

Cumulative return: 589.89%

Annualized Sharpe: 0.74

Daily mean return: 0.0433%

Daily volatility: 0.9290%

7. Performance Interpretation

The leveraged regime strategy demonstrates:

Higher Sharpe ratio compared to buy-and-hold

Significantly reduced drawdowns

Competitive total returns

More stable equity growth during stress periods (e.g., 2008, 2020)

These results indicate that the Transformer model learns regimes that are economically meaningful and suitable for dynamic risk allocation.

8. How to Run
Training
python -m deepregime.training.train_transformer

Inference
python -m deepregime.inference

Backtesting
python -m deepregime.backtest.backtest_leverage

9. Future Work

Predict regime transitions (multi-step forecasting)

Multi-asset regime modeling (SPY, VIX, TLT, GOLD, credit spreads)

Volatility targeting and portfolio optimization

Incorporation of transaction costs and slippage

Hyperparameter tuning and walk-forward validation

Hidden Markov Model baseline comparison
