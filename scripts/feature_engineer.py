#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from talib import RSI, BBANDS, MACD, ATR
from pathlib import Path
import logging
from typing import List, Optional, Tuple, Dict, Union


class StockFeatureEngineer:
    """
    A class for feature engineering on stock price data.
    
    This class handles:
    - Loading and preprocessing stock data
    - Computing technical indicators (RSI, Bollinger Bands, MACD, ATR, etc.)
    - Creating return features and forward returns
    - Universe selection and filtering
    - Data cleaning and outlier removal
    """
    
    def __init__(self, 
                 data_store_path: str = './data/nasdaq_data_yf.h5',
                 start_date: str = '2009-01-01',
                 end_date: str = '2025-04-25',
                 min_years: int = 7,
                 top_n_stocks: int = 1000):
        """
        Initialize the StockFeatureEngineer.
        
        Args:
            data_store_path (str): Path to the HDF5 data store
            start_date (str): Start date for data selection
            end_date (str): End date for data selection
            min_years (int): Minimum years of data required per stock
            top_n_stocks (int): Number of top stocks by market cap to include
        """
        self.data_store_path = Path(data_store_path)
        self.start_date = start_date
        self.end_date = end_date
        self.min_years = min_years
        self.top_n_stocks = top_n_stocks
        
        # Constants
        self.MONTH = 21
        self.YEAR = 12 * self.MONTH
        self.T = [1, 5, 10, 21, 42, 63]  # Time periods for returns
        self.OHLCV = ['open', 'close', 'low', 'high', 'volume']
        
        # Data containers (dataframes)
        self.prices = None
        self.metadata = None
        
        # logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # plotting style
        sns.set_style('darkgrid')
        self.idx = pd.IndexSlice
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load price and metadata from HDF5 store.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: prices and metadata DataFrames
        """
        try:
            with pd.HDFStore(self.data_store_path) as store:
                prices = (store['nasdaq/price']
                         .loc[self.idx[self.start_date:self.end_date, :], self.OHLCV]
                         .swaplevel()
                         .sort_index())
                
                metadata = store['nasdaq/metadata'].loc[:, ['marketcap', 'sector']]
            
            # Adjust volume and set proper index names
            prices.volume /= 1e3  # make vol figures smaller
            prices.index.names = ['symbol', 'date']
            metadata.index.name = 'symbol'
            
            self.logger.info(f"Loaded data: {len(prices)} price records, {len(metadata)} metadata records")
            return prices, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def filter_by_observations(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Filter stocks that have insufficient observations.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Filtered price data
        """
        min_obs = self.min_years * self.YEAR
        nobs = prices.groupby(level='symbol').size()
        keep = nobs[nobs > min_obs].index
        
        filtered_prices = prices.loc[self.idx[keep, :], :]
        self.logger.info(f"Filtered to {len(keep)} stocks with ≥{self.min_years} years of data")
        
        return filtered_prices
    
    def align_price_metadata(self, prices: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align price and metadata by ensuring both have the same symbols.
        
        Args:
            prices (pd.DataFrame): Price data
            metadata (pd.DataFrame): Metadata
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned prices and metadata
        """
        # Clean metadata
        metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
        metadata.sector = metadata.sector.str.lower().str.replace(' ', '_')
        
        # Find common symbols
        shared = (prices.index.get_level_values('symbol').unique()
                 .intersection(metadata.index))
        
        metadata = metadata.loc[shared, :]
        prices = prices.loc[self.idx[shared, :], :]
        
        self.logger.info(f"Aligned data: {len(shared)} common symbols")
        return prices, metadata
    
    def select_universe(self, prices: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select universe based on market cap.
        
        Args:
            prices (pd.DataFrame): Price data
            metadata (pd.DataFrame): Metadata
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Filtered prices and metadata
        """
        universe = metadata.marketcap.astype(float).nlargest(self.top_n_stocks).index
        prices = prices.loc[self.idx[universe, :], :]
        metadata = metadata.loc[universe]
        
        self.logger.info(f"Selected universe: top {self.top_n_stocks} stocks by market cap")
        return prices, metadata
    
    def compute_dollar_volume_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dollar volume and ranking features.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with dollar volume features
        """
        # Compute dollar volume
        prices = prices.copy()
        prices['dollar_vol'] = prices[['close', 'volume']].prod(1).div(1e3)
        
        # 21-day moving average
        dollar_vol_ma = (prices
                        .dollar_vol
                        .unstack('symbol')
                        .rolling(window=21, min_periods=1)
                        .mean())
        
        # Rank stocks by moving average
        prices['dollar_vol_rank'] = (dollar_vol_ma
                                   .rank(axis=1, ascending=False)
                                   .stack('symbol')
                                   .swaplevel())
        
        self.logger.info("Computed dollar volume features")
        return prices
    
    def apply_rsi(self, group: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Helper function to apply RSI to a group."""
        return pd.Series(RSI(group.values, timeperiod=timeperiod), index=group.index)
    
    def compute_rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Relative Strength Index.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with RSI feature
        """
        prices = prices.copy()
        prices['rsi'] = (prices.groupby(level='symbol', group_keys=False)
                        .close.apply(self.apply_rsi))
        
        self.logger.info("Computed RSI")
        return prices
    
    def compute_bb(self, close: pd.Series) -> pd.DataFrame:
        """Helper function to compute Bollinger Bands."""
        high, mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)
    
    def compute_bollinger_bands(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bands features.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with Bollinger Bands features
        """
        prices = prices.copy()
        
        # Compute raw Bollinger Bands
        bb_data = (prices.groupby(level='symbol', group_keys=False)
                  .close.apply(self.compute_bb))
        prices = prices.join(bb_data)
        
        # Transform to relative measures
        prices['bb_high'] = (prices.bb_high.sub(prices.close)
                           .div(prices.bb_high).apply(np.log1p))
        prices['bb_low'] = (prices.close.sub(prices.bb_low)
                          .div(prices.close).apply(np.log1p))
        
        self.logger.info("Computed Bollinger Bands")
        return prices
    
    def compute_atr_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Average True Range features.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with ATR features
        """
        prices = prices.copy()
        
        # Normalized ATR
        prices['NATR'] = prices.groupby(level='symbol', group_keys=False).apply(
            lambda x: talib.NATR(x.high, x.low, x.close))
        
        # Standardized ATR
        def compute_atr_std(stock_data):
            df = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
            return df.sub(df.mean()).div(df.std())
        
        prices['ATR'] = (prices.groupby('symbol', group_keys=False)
                        .apply(compute_atr_std))
        
        self.logger.info("Computed ATR features")
        return prices
    
    def compute_macd_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD-related features.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with MACD features
        """
        prices = prices.copy()
        
        # PPO (Percentage Price Oscillator)
        prices['PPO'] = (prices.groupby(level='symbol', group_keys=False)
                        .close.apply(talib.PPO))
        
        # Standardized MACD
        def compute_macd_std(close):
            macd = MACD(close)[0]
            return (macd - np.mean(macd)) / np.std(macd)
        
        prices['MACD'] = (prices.groupby('symbol', group_keys=False)
                         .close.apply(compute_macd_std))
        
        self.logger.info("Computed MACD features")
        return prices
    
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical returns for different time periods.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with return features
        """
        prices = prices.copy()
        by_sym = prices.groupby(level='symbol').close
        
        for t in self.T:
            prices[f'r{t:02}'] = by_sym.pct_change(t)
        
        self.logger.info(f"Computed returns for periods: {self.T}")
        return prices
    
    def compute_return_deciles(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily historical return deciles.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with return decile features
        """
        prices = prices.copy()
        
        for t in self.T:
            prices[f'r{t:02}dec'] = (prices[f'r{t:02}']
                                   .groupby(level='date', group_keys=False)
                                   .apply(lambda x: pd.qcut(x, q=10, labels=False, 
                                                          duplicates='drop')))
        
        self.logger.info("Computed return deciles")
        return prices
    
    def compute_sector_return_quintiles(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily sector return quintiles.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with sector return quintile features
        """
        prices = prices.copy()
        
        for t in self.T:
            prices[f'r{t:02}q_sector'] = (prices
                                        .groupby(['date', 'sector'])[f'r{t:02}']
                                        .transform(lambda x: pd.qcut(x, q=5, labels=False,
                                                                   duplicates='drop')))
        
        self.logger.info("Computed sector return quintiles")
        return prices
    
    def compute_forward_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward returns for prediction targets.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with forward return features
        """
        prices = prices.copy()
        
        for t in [1, 5, 21]:
            prices[f'r{t:02}_fwd'] = (prices.groupby(level='symbol')[f'r{t:02}']
                                    .shift(-t))
        
        self.logger.info("Computed forward returns")
        return prices
    
    def remove_outliers(self, prices: pd.DataFrame, return_threshold: float = 1.0) -> pd.DataFrame:
        """
        Remove outliers based on daily returns.
        
        Args:
            prices (pd.DataFrame): Price data
            return_threshold (float): Threshold for daily returns (100% = 1.0)
            
        Returns:
            pd.DataFrame: Prices with outliers removed
        """
        outliers = prices[prices.r01 > return_threshold].index.get_level_values('symbol').unique()
        prices = prices.drop(outliers, level='symbol')
        
        self.logger.info(f"Removed {len(outliers)} outlier stocks with daily returns > {return_threshold*100}%")
        return prices
    
    def add_time_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Prices with time features
        """
        prices = prices.copy()
        prices['year'] = prices.index.get_level_values('date').year
        prices['month'] = prices.index.get_level_values('date').month
        prices['weekday'] = prices.index.get_level_values('date').weekday
        
        self.logger.info("Added time features")
        return prices

    
    def add_sector_encoding(self, prices: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Add sector encoding to prices.
        
        Args:
            prices (pd.DataFrame): Price data
            metadata (pd.DataFrame): Metadata with sector information
            
        Returns:
            pd.DataFrame: Prices with sector encoding
        """
        prices = prices.copy()
        metadata = metadata.copy()
        metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
        prices = prices.join(metadata[['sector']])
        
        self.logger.info("Added sector encoding")
        return prices
    
    def plot_rsi_distribution(self, prices: pd.DataFrame) -> None:
        """Plot RSI distribution with signal thresholds."""
        ax = sns.distplot(prices.rsi.dropna())
        ax.axvline(30, ls='--', lw=1, c='k')
        ax.axvline(70, ls='--', lw=1, c='k')
        ax.set_title('RSI Distribution with Signal Threshold')
        sns.despine()
        plt.tight_layout()
        plt.show()
    
    def plot_bollinger_bands_distribution(self, prices: pd.DataFrame) -> None:
        """Plot Bollinger Bands distribution for top stocks."""
        fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
        top_stocks = prices.dollar_vol_rank < 100
        
        sns.distplot(prices.loc[top_stocks, 'bb_low'].dropna(), ax=axes[0])
        axes[0].set_title('BB Low Distribution (Top 100 Stocks)')
        
        sns.distplot(prices.loc[top_stocks, 'bb_high'].dropna(), ax=axes[1])
        axes[1].set_title('BB High Distribution (Top 100 Stocks)')
        
        sns.despine()
        plt.tight_layout()
        plt.show()
    
    def get_feature_summary(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for return features.
        
        Args:
            prices (pd.DataFrame): Price data with features
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        #return prices[[f'r{t:02}' for t in self.T]].describe()
        return prices.info()
    
    def save_processed_data(self, prices: pd.DataFrame, output_path: str = 'data_yf.h5') -> None:
        """
        Save processed data to HDF5, excluding raw OHLCV data.
        
        Args:
            prices (pd.DataFrame): Processed price data
            output_path (str): Output file path
        """
        # Remove raw OHLCV columns
        model_data = prices.drop(['open', 'close', 'low', 'high', 'volume'], axis=1)
        model_data.to_hdf(output_path, 'model_data', mode='w')
        
        self.logger.info(f"Saved processed data to {output_path}")
        self.logger.info(f"Final dataset shape: {model_data.shape}")
        self.logger.info(f"Features: {list(model_data.columns)}")
    
    def run_full_pipeline(self, save_output: bool = True, output_path: str = 'data_yf.h5') -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            save_output (bool): Whether to save the processed data
            output_path (str): Output file path if saving
            
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        self.logger.info("Starting feature engineering pipeline...")
        
        # Load data
        prices, metadata = self.load_data()
        
        # Data filtering and alignment
        prices = self.filter_by_observations(prices)
        prices, metadata = self.align_price_metadata(prices, metadata)
        prices, metadata = self.select_universe(prices, metadata)
        
        # Compute features
        prices = self.compute_dollar_volume_features(prices)
        prices = self.compute_rsi(prices)
        prices = self.compute_bollinger_bands(prices)
        prices = self.compute_atr_features(prices)
        prices = self.compute_macd_features(prices)
        
        # Add sector information
        prices = self.add_sector_encoding(prices, metadata)
        
        # Compute returns and related features
        prices = self.compute_returns(prices)
        prices = self.compute_return_deciles(prices)
        prices = self.compute_sector_return_quintiles(prices)
        prices = self.compute_forward_returns(prices)
        
        # Clean data
        prices = self.remove_outliers(prices)
        prices = self.add_time_features(prices)
        
        # Store processed data
        self.prices = prices
        self.metadata = metadata
        
        # Save if requested
        if save_output:
            self.save_processed_data(prices, output_path)
        
        self.logger.info("Feature engineering pipeline completed!")
        return prices

