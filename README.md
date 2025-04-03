# Stock Price Movement Prediction with PyTorch

This project uses a simple machine learning pipeline in PyTorch to predict **next-day stock price movement** (up or down) based on historical OHLCV data and technical indicators.

The model is clearly overfitting at the moment. This is somewhat expected since the model is too simple and not using enough data.
## What It Does

- Downloads historical stock data (e.g., AAPL, MSFT) using `yfinance`
- Calculates features such as:
  - Technical indicators (RSI, MACD, SMA, ATR, etc.)
- Trains a binary classification model (MLP) to predict:
  - Will the stock price go **up** or **down** the next day?

## Future Improvements

- Backtesting simple trading strategies using model predictions
- Try LSTM?
- Enhanced feature engineering (e.g., cross-asset signals)
- Hyperparameter tuning

## Getting Started

```bash
pip install -r requirements.txt
