import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    def __init__(self, ticker_file_path: str, output_dir: str):
        """
        Initialize the Historical Data Downloader
        
        Args:
            ticker_file_path (str): Path to the CSV file containing tickers
            output_dir (str): Directory to save the price data CSV files
        """
        self.ticker_file_path = ticker_file_path
        self.output_dir = output_dir
        self.timeframes = {
            '1m': '1m',
            '5m': '5m', 
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # Maximum periods for different timeframes (yfinance limitations)
        self.max_periods = {
            '1m': '7d',    # 1-minute data: max 7 days
            '5m': '60d',   # 5-minute data: max 60 days
            '30m': '60d',  # 30-minute data: max 60 days
            '1h': '2y',    # 1-hour data: max 2 years
            '4h': '2y',    # 4-hour data: max 2 years
            '1d': 'max'    # 1-day data: max available (usually 10+ years)
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_tickers(self) -> List[str]:
        """
        Load ticker symbols from the CSV file
        
        Returns:
            List[str]: List of ticker symbols
        """
        try:
            df = pd.read_csv(self.ticker_file_path)
            tickers = df['Symbol'].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {self.ticker_file_path}")
            return tickers
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []
    
    def get_market_hours_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Get historical data for a ticker during market hours only
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period for data (e.g., '7d', '60d')
            interval (str): Data interval (e.g., '1m', '5m')
            
        Returns:
            pd.DataFrame: Historical price data during market hours
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Download data
            data = stock.history(period=period, interval=interval, prepost=False)
            
            if data.empty:
                logger.warning(f"No data available for {ticker} with interval {interval}")
                return pd.DataFrame()
            
            # Filter for market hours (9:30 AM to 4:00 PM EST)
            # yfinance data is already filtered for regular market hours when prepost=False
            
            # Reset index to make datetime a column
            data = data.reset_index()
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Reorder columns
            cols = ['Ticker', 'Datetime'] + [col for col in data.columns if col not in ['Ticker', 'Datetime']]
            data = data[cols]
            
            logger.info(f"Retrieved {len(data)} records for {ticker} ({interval})")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker} ({interval}): {e}")
            return pd.DataFrame()
    
    def save_data_by_date(self, data: pd.DataFrame, ticker: str, interval: str):
        """
        Save data to CSV files organized by date
        
        Args:
            data (pd.DataFrame): Price data
            ticker (str): Stock ticker symbol
            interval (str): Data interval
        """
        if data.empty:
            return
        
        try:
            # Convert Datetime column to datetime if it's not already
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Group data by date
            data['Date'] = data['Datetime'].dt.date
            
            for date, day_data in data.groupby('Date'):
                # Create filename with ticker, date, and interval
                filename = f"{ticker}_{date}_{interval}.csv"
                filepath = os.path.join(self.output_dir, filename)
                
                # Drop the Date column before saving
                day_data_clean = day_data.drop('Date', axis=1)
                
                # Save to CSV
                day_data_clean.to_csv(filepath, index=False)
                logger.info(f"Saved {len(day_data_clean)} records to {filename}")
                
        except Exception as e:
            logger.error(f"Error saving data for {ticker} ({interval}): {e}")
    
    def download_ticker_data(self, ticker: str):
        """
        Download all timeframe data for a single ticker
        
        Args:
            ticker (str): Stock ticker symbol
        """
        logger.info(f"Starting download for {ticker}")
        
        for interval_name, interval_code in self.timeframes.items():
            try:
                period = self.max_periods[interval_code]
                logger.info(f"Downloading {ticker} - {interval_name} data (period: {period})")
                
                # Get data
                data = self.get_market_hours_data(ticker, period, interval_code)
                
                if not data.empty:
                    # Save data by date
                    self.save_data_by_date(data, ticker, interval_name)
                else:
                    logger.warning(f"No data retrieved for {ticker} - {interval_name}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {ticker} - {interval_name}: {e}")
                continue
    
    def download_all_tickers(self):
        """
        Download historical data for all tickers
        """
        tickers = self.load_tickers()
        
        if not tickers:
            logger.error("No tickers loaded. Exiting.")
            return
        
        total_tickers = len(tickers)
        logger.info(f"Starting download for {total_tickers} tickers")
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"Processing ticker {i}/{total_tickers}: {ticker}")
                self.download_ticker_data(ticker)
                
                # Longer delay between tickers to avoid rate limiting
                if i < total_tickers:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        logger.info("Download completed for all tickers")

def main():
    """
    Main function to run the historical data download
    """
    # Configuration
    ticker_file = "data/comprehensive_ticker_list.csv"
    output_directory = "data/price_data"
    
    # Initialize downloader
    downloader = HistoricalDataDownloader(ticker_file, output_directory)
    
    # Start download
    logger.info("=== Historical Data Download Started ===")
    start_time = datetime.now()
    
    downloader.download_all_tickers()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"=== Download Completed in {duration} ===")

if __name__ == "__main__":
    main()
