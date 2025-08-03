#!/usr/bin/env python3
"""
Script to fetch daily price data from market open to close for all tickers
in the comprehensive_ticker_list.csv file and save each ticker's data as a separate CSV.
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import time
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tickers(csv_path: str) -> List[str]:
    """
    Load tickers from the comprehensive ticker list CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing tickers
        
    Returns:
        List[str]: List of ticker symbols
    """
    try:
        df = pd.read_csv(csv_path)
        tickers = df['Symbol'].tolist()
        logger.info(f"Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from {csv_path}: {e}")
        return []

def fetch_daily_data(ticker: str, period: str = "1d", interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Fetch intraday price data for a single ticker from market open to close.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Period to fetch data for (default: "1d" for today)
        interval (str): Data interval (default: "1m" for 1-minute intervals)
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with price data or None if failed
    """
    try:
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch intraday data
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data available for ticker: {ticker}")
            return None
            
        # Reset index to make datetime a column
        data.reset_index(inplace=True)
        
        # Add ticker symbol as a column
        data['Symbol'] = ticker
        
        logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def save_ticker_data(data: pd.DataFrame, ticker: str, output_dir: str, date_str: str) -> bool:
    """
    Save ticker data to a CSV file.
    
    Args:
        data (pd.DataFrame): Price data for the ticker
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save the file
        date_str (str): Date string to include in filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with date
        filename = f"{ticker}_{date_str}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        data.to_csv(filepath, index=False)
        logger.info(f"Saved data for {ticker} to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")
        return False

def main():
    """
    Main function to fetch and save price data for all tickers.
    """
    # Configuration
    ticker_csv_path = "data/comprehensive_ticker_list.csv"
    output_dir = "data/price_data"
    
    # Get today's date for filename
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    
    logger.info(f"Starting price data collection for {date_str}")
    
    # Load tickers
    tickers = load_tickers(ticker_csv_path)
    if not tickers:
        logger.error("No tickers loaded. Exiting.")
        return
    
    # Track statistics
    successful_downloads = 0
    failed_downloads = 0
    
    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
        
        # Fetch data
        data = fetch_daily_data(ticker)
        
        if data is not None and not data.empty:
            # Save data
            if save_ticker_data(data, ticker, output_dir, date_str):
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            failed_downloads += 1
            logger.warning(f"No data available for {ticker}")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Log progress every 50 tickers
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
    
    # Final statistics
    logger.info(f"Data collection completed!")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Total tickers processed: {len(tickers)}")
    logger.info(f"Success rate: {(successful_downloads/len(tickers)*100):.1f}%")

if __name__ == "__main__":
    main()