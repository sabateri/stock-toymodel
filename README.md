# Stock Price Movement Prediction with PyTorch

This project uses a simple machine learning pipeline in PyTorch to predict 1-day forward returns based on historical data and technical indicators.

This project is for educational purposes only.

## What It Does

- Downloads historical stock data
- Calculates features such as:
  - Technical indicators (RSI, MACD, SMA, ATR, etc.)
- Trains a Neural Network (MLP) to predict forward returns

## Future Improvements

- Backtesting double-check using backtrader
- Enhanced feature engineering (e.g., cross-asset signals)
- Hyperparameter tuning

## Getting Started

```bash
pip install -r requirements.txt
