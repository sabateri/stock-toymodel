# Stock Price Movement Prediction with PyTorch

This project implements a simple deep learning pipeline to predict short-term stock price movements using historical data and technical indicators. A  neural network is trained on engineered features like RSI, MACD, and SMA to forecast 1-day forward returns. The model is then evaluated via realistic backtesting using Zipline-Reloaded and Backtrader.

This project is for educational purposes only.

The following image shows the output of the backtest using zipline, using a simple Neural Network with two hidden layers, with 16 and 8 neurons, respectively.
![arxiv webpage image](images/backtest_zipline_returns.png)
## What It Does

- Downloads historical stock data
- Calculates features such as:
  - Technical indicators (RSI, MACD, SMA, ATR, etc.)
- Trains a Neural Network to predict forward returns
- Uses zipline or backtrader to backtest the results with historical data

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/sabateri/stock-toymodel.git
   cd stock-toymodel
   ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Backtesting
**Option 1)** [Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded.git)

Since zipline is deprecated, other package versions are needed, to run backtests using zipline-reloaded (a maintained version of zipline) do
```bash
pip install -r requirements_zipline.txt
```

**Option 2)** [Backtrader](https://www.backtrader.com/)
```bash
pip install -r requirements_backtrader.txt
```

The steps to run are

1. ```notebooks/create_datasets.ipynb``` creates the datasets needed
2. ```notebooks/feature_engineering.ipynb``` engineers features for the model
3. ```notebooks/optimizing_NN_pytorch.ipynb``` trains NN
4. Backtesting notebooks:  
   **4a.** `notebooks/backtest_zipline.ipynb` — backtests using Zipline  
   **4b.** `notebooks/backtest_backtrader.ipynb` — backtests using Backtrader 

<!--
## Data Sources
- [US Stock Symbols GitHub Repo](https://github.com/rreichel3/US-Stock-Symbols.git): Used to obtain lists of publicly traded tickers.
- Price data fetched via `yfinance`
!-->

## Future Improvements

- Incorporate more robust features
- Add hyperparameter tuning
- Integrate paper trading to test strategies in live, non-risk settings (IBKR)