# Stock Price Movement Prediction with PyTorch

This project implements a simple deep learning pipeline to predict short-term stock price movements using historical data and technical indicators. A  neural network and an XGBoost model are trained on engineered features like RSI, MACD, and SMA to forecast 1-day forward returns. The model is then evaluated via realistic backtesting using Backtrader.

This project is for educational purposes only.

The following image shows the output of the backtest, using a simple Neural Network with two hidden layers, with 16 and 8 neurons, respectively.
![arxiv webpage image](images/backtest_zipline_returns.png)
## What It Does

- Downloads historical stock data
- Calculates features such as:
  - Technical indicators (RSI, MACD, SMA, ATR, etc.)
- Trains a XGBoost/Neural Network to predict forward returns
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
Steps 1-3 are already automatized in the ```main.py``` script. It creates the dataset and engines features. In this case running 16 years of data
   ```
   python3 main.py --years 16 --start_date 2009-01-01 --end_date 2025-01-01  --feature-engineer
   ```

This will create all the samples with features needed for the analysis. One can then run all the notebooks used for training, testing, evaluation and backtesting

4. ```04_optimizing_xgboost.ipynb``` -> train and tune XGBoost

   (4.1 ```notebooks/optimizing_NN_pytorch.ipynb``` -> train the NN)
5. ```05_evaluate_xgboost.ipynb``` -> evaluates the results
6. ```06_model_interpretation.ipynb``` -> model interpretation
7. ```07_making_out_of_sample_predictions.ipynb``` -> out-of-sample predictions
8. ```08_backtest_backtrader.ipynb``` -> backtest with backtrader

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




## Future Improvements
- Include partial dependence plots in the model interpretation
- Incorporate more robust features
- Integrate paper trading to test strategies in live, non-risk settings (IBKR)
