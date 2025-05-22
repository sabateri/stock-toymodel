from scripts.data_fetcher import StockDataFetcher
from scripts.feature_engineer import StockFeatureEngineer
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data fetching and feature engineering configuration')
    
    # Boolean flags for data fetching and feature engineering
    parser.add_argument('--fetch-data', '--fetch_data', 
                       action='store_true', 
                       default=True,
                       help='Whether to fetch data (default: True)')
    parser.add_argument('--no-fetch-data', '--no_fetch_data',
                       dest='fetch_data',
                       action='store_false',
                       help='Disable data fetching')
    
    parser.add_argument('--feature-engineer', '--feature_engineer',
                       action='store_true',
                       default=False,
                       help='Whether to create features (default: False)')
    
    # Number of years for data fetching
    parser.add_argument('--years', '-y',
                       type=int,
                       default=16,
                       help='Number of years (starting from yesterday) to fetch data from (default: 16)')
    
    # Date range for feature creation
    parser.add_argument('--start-date', '--start_date',
                       type=str,
                       default='2009-01-01',
                       help='Start date for feature creation (format: YYYY-MM-DD, default: 2009-01-01)')
    
    parser.add_argument('--end-date', '--end_date',
                       type=str,
                       default='2025-01-01',
                       help='End date for feature creation (format: YYYY-MM-DD, default: 2025-01-01)')
    
    # Data storage location
    parser.add_argument('--data-store', '--data_store',
                       type=str,
                       default='./data',
                       help='Directory to store data (default: ./data)')
    
    # Minimum years filter
    parser.add_argument('--min-years', '--min_years',
                       type=int,
                       default=7,
                       help='Filter tickers with at least this many years of data (default: 7)')
    
    return parser.parse_args()

def main():

    # Parse arguments
    args = parse_arguments()
    
    # Access the variables
    FETCH_DATA = args.fetch_data
    FEATURE_ENGINEER = args.feature_engineer
    YEARS = args.years
    START_DATE = args.start_date
    END_DATE = args.end_date
    DATA_STORE = args.data_store
    MIN_YEARS = args.min_years
    
    # Print configuration
    print("Configuration:")
    print(f"FETCH_DATA = {FETCH_DATA}")
    print(f"FEATURE_ENGINEER = {FEATURE_ENGINEER}")
    print(f"YEARS = {YEARS}")
    print(f"START_DATE = '{START_DATE}'")
    print(f"END_DATE = '{END_DATE}'")
    print(f"DATA_STORE = '{DATA_STORE}'")
    print(f"MIN_YEARS = {MIN_YEARS}")


    # initialize the data fetcher
    fetcher = StockDataFetcher()

    # fetch and store NASDAQ data. If no end_date is provided, it will fetch until yesterday
    if FETCH_DATA:
        fetcher.fetch_and_store_nasdaq_data(years=YEARS, end_date=END_DATE)
        #fetcher.fetch_and_store_nasdaq_data(years=YEARS, end_date=None)

    feature_engineer = StockFeatureEngineer(
    data_store_path=f'{DATA_STORE}/nasdaq_data_yf.h5',
    start_date=START_DATE,
    end_date=END_DATE,
    min_years=MIN_YEARS, 
    top_n_stocks=1000
)

    # run the complete pipeline to extract features, which are used then for training
    if FEATURE_ENGINEER:
        processed_data = feature_engineer.run_full_pipeline(save_output=True,output_path=f'{DATA_STORE}/data_features.h5')

        # get info of the features dataframe
        print("\nReturn features summary:")
        print(feature_engineer.get_feature_summary(processed_data))


if __name__ == "__main__":
    main()
