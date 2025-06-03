#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import urllib3
from typing import List, Optional, Union


class StockDataFetcher:
    """
    A class to fetch and store stock market data using yfinance.
    
    Supports fetching data for S&P 500 and NASDAQ tickers, with automatic
    data storage to HDF5 format.
    """
    
    def __init__(self, data_store_path: str = './data/nasdaq_data_yf_test.h5'):
        """
        Initialize the StockDataFetcher.
        
        Args:
            data_store_path (str): Path to the HDF5 data store file
        """
        self.data_store = Path(data_store_path)
        self.data_store.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress urllib3 connection pool warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    
    def get_nasdaq_tickers(self, json_path: str = './data/nasdaq_tickers.json') -> List[str]:
        """
        Load NASDAQ ticker symbols from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file containing NASDAQ tickers
            
        Returns:
            List[str]: List of NASDAQ ticker symbols
        """
        try:
            nasdaq_tickers = pd.read_json(json_path)
            nasdaq_ticker_list = nasdaq_tickers[0].str.replace('.', '-').unique().tolist()
            self.logger.info(f"Retrieved {len(nasdaq_ticker_list)} NASDAQ tickers")
            return nasdaq_ticker_list
        except Exception as e:
            self.logger.error(f"Error loading NASDAQ tickers from {json_path}: {e}")
            raise
    
    def load_data(self, 
                  tickers = 'AAPL', 
                  end_date: str = '2024-01-01', 
                  years: int = 8) -> pd.DataFrame:
        """
        Download stock data using yfinance.
        
        Args:
            tickers (Union[str, List[str]]): Ticker symbol(s) to download
            end_date (str): End date for data download (YYYY-MM-DD format)
            years (int): Number of years of historical data to fetch
            
        Returns:
            pd.DataFrame: Downloaded stock data
        """
        try:
            start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * years)
            self.logger.info(f"Downloading data from {start_date.date()} to {end_date}")
            
            df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                self.logger.warning("No data downloaded")
                return df
            
            self.logger.info(f"Downloaded data shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            raise
    
    def process_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw price data into the desired format.
        
        Args:
            df (pd.DataFrame): Raw price data from yfinance
            
        Returns:
            pd.DataFrame: Processed price data with proper indexing
        """
        try:
            # Stack the data and rename columns
            prices = df.stack().rename(columns=str.lower)
            prices.index.names = ['date', 'ticker']
            
            # Drop 'close' column and rename 'adj close' to 'close'
            if 'close' in prices.columns:
                prices = prices.drop('close', axis=1)
            if 'adj close' in prices.columns:
                prices.rename(columns={'adj close': 'close'}, inplace=True)
            
            self.logger.info(f"Processed price data shape: {prices.shape}")
            return prices
            
        except Exception as e:
            self.logger.error(f"Error processing price data: {e}")
            raise
    
    def load_metadata(self, json_path: str = './data/nasdaq_full_tickers.json') -> pd.DataFrame:
        """
        Load and process metadata from JSON file.
        
        Args:
            json_path (str): Path to the metadata JSON file
            
        Returns:
            pd.DataFrame: Processed metadata with ticker as index
        """
        try:
            metadata = pd.read_json(json_path).rename(columns={'symbol': 'ticker'})
            metadata['ticker'] = metadata['ticker'].str.replace('.', '-')
            metadata = metadata.drop_duplicates(subset=['ticker']).set_index('ticker')
            metadata.columns = metadata.columns.str.lower()
            
            self.logger.info(f"Loaded metadata for {len(metadata)} tickers")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading metadata from {json_path}: {e}")
            raise
    
    def save_to_hdf5(self, 
                     prices: pd.DataFrame, 
                     metadata: Optional[pd.DataFrame] = None,
                     price_key: str = 'nasdaq/price',
                     metadata_key: str = 'nasdaq/metadata') -> None:
        """
        Save price data and metadata to HDF5 store.
        
        Args:
            prices (pd.DataFrame): Price data to save
            metadata (Optional[pd.DataFrame]): Metadata to save
            price_key (str): HDF5 key for price data
            metadata_key (str): HDF5 key for metadata
        """
        try:
            with pd.HDFStore(self.data_store, mode='w') as store:
                store.put(price_key, prices)
                self.logger.info(f"Saved price data to {self.data_store}")
                
                if metadata is not None:
                    store.put(metadata_key, metadata)
                    self.logger.info(f"Saved metadata to {self.data_store}")
                    
        except Exception as e:
            self.logger.error(f"Error saving to HDF5: {e}")
            raise
    
    def fetch_and_store_nasdaq_data(self, 
                                    years: int = 16,
                                    end_date: Optional[str] = None) -> None:
        """
        Complete workflow to fetch and store NASDAQ data.
        
        Args:
            years (int): Number of years of historical data to fetch
            end_date (Optional[str]): End date for data (defaults to yesterday)
        """
        try:
            # Set end_date to yesterday if not provided
            if end_date is None:
                end_date = (datetime.today() - pd.DateOffset(1)).strftime('%Y-%m-%d')
            

            self.logger.info("Getting the nasdaq tickers...")
            # Get NASDAQ tickers
            nasdaq_tickers = self.get_nasdaq_tickers()

            self.logger.info("Downloading data from yf...")
            # Download price data
            raw_data = self.load_data(tickers=nasdaq_tickers, end_date=end_date, years=years)
            
            self.logger.info("Processing data...")
            # Process price data
            prices = self.process_price_data(raw_data)

            self.logger.info("Loading metadata...")
            # Load metadata
            metadata = self.load_metadata()
            
            self.logger.info("Saving the data to h5 file...")
            # Save to HDF5
            self.save_to_hdf5(prices, metadata)
            
            self.logger.info("NASDAQ data fetch and store completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in fetch_and_store_nasdaq_data: {e}")
            raise
    
