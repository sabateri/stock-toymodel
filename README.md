# Stock Price Movement Prediction with PyTorch

This project implements a simple deep learning pipeline to predict short-term stock price movements using historical data and technical indicators. A  neural network is trained on engineered features like RSI, MACD, and SMA to forecast 1-day forward returns. The model is then evaluated via realistic backtesting using Backtrader.

This project is for educational purposes only.

The following image shows the output of the backtest, using a simple Neural Network with two hidden layers, with 16 and 8 neurons, respectively.
![arxiv webpage image](images/backtest_zipline_returns.png)
## What It Does

- Downloads historical stock data
- Calculates features such as:
  - Technical indicators (RSI, MACD, SMA, ATR, etc.)
- Trains a Neural Network to predict forward returns
- Uses backtrader to backtest the results with historical data

## Getting Started
### Prequisites
Make sure you have Python ```3.11.11``` installed.

1. Clone this repository:
   ```bash
   git clone https://github.com/sabateri/stock-toymodel.git
   cd stock-toymodel
   ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```



### Steps to run the code
1. Create the dataset and engine features. In this case running 16 years of data
   ```
   python3 main.py --years 16 --start_date 2009-01-01 --end_date 2025-01-01  --feature-engineer
   ```
2. ```notebooks/optimizing_NN_pytorch.ipynb``` -> train the NN

<!--
## Data Sources
- [US Stock Symbols GitHub Repo](https://github.com/rreichel3/US-Stock-Symbols.git): Used to obtain lists of publicly traded tickers.
- Price data fetched via `yfinance`
!-->

### Backtesting
<!-- **Option 1)** [Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded.git)

Since zipline is deprecated, other package versions are needed, to run backtests using zipline-reloaded (a maintained version of zipline) do
```bash
pip install -r requirements_zipline.txt
```

**Option 2)** 
 -->
We backtest using [Backtrader](https://www.backtrader.com/), I recommend to run this part from another environment, since it uses different packages versions
```bash
pip install -r requirements_backtrader.txt
```
3. Backtesting notebook:  
   <!-- **4a.** `notebooks/backtest_zipline.ipynb` — backtests using Zipline   -->
    `notebooks/backtest_backtrader.ipynb` -> backtests using Backtrader 



## Future Improvements
- Tune the NN and move step 3. to more production ready
- Incorporate more robust features
- Add hyperparameter tuning
- Integrate paper trading to test strategies in live, non-risk settings (IBKR)