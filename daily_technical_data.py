#!/usr/bin/env python3
"""
Script to fetch technical data (dividends, splits, actions, cap gains, shares full, info, and news)
for all tickers in the comprehensive_ticker_list.csv file and save each ticker's data as separate CSVs.
"""

import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
import time
import logging
from typing import List, Optional, Dict, Any

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

def fetch_technical_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch technical data for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Dict[str, Any]: Dictionary containing all technical data
    """
    data = {
        'ticker': ticker,
        'dividends': None,
        'splits': None,
        'actions': None,
        'capital_gains': None,
        'shares_full': None,
        'info': None,
        'news': None,
        'fetch_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch dividends
        try:
            dividends = stock.dividends
            if not dividends.empty:
                dividends_df = dividends.reset_index()
                dividends_df['Symbol'] = ticker
                data['dividends'] = dividends_df
                logger.info(f"Fetched {len(dividends_df)} dividend records for {ticker}")
            else:
                logger.info(f"No dividend data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching dividends for {ticker}: {e}")
        
        # Fetch splits
        try:
            splits = stock.splits
            if not splits.empty:
                splits_df = splits.reset_index()
                splits_df['Symbol'] = ticker
                data['splits'] = splits_df
                logger.info(f"Fetched {len(splits_df)} split records for {ticker}")
            else:
                logger.info(f"No split data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching splits for {ticker}: {e}")
        
        # Fetch actions (combined dividends and splits)
        try:
            actions = stock.actions
            if not actions.empty:
                actions_df = actions.reset_index()
                actions_df['Symbol'] = ticker
                data['actions'] = actions_df
                logger.info(f"Fetched {len(actions_df)} action records for {ticker}")
            else:
                logger.info(f"No action data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching actions for {ticker}: {e}")
        
        # Fetch capital gains
        try:
            capital_gains = stock.capital_gains
            if not capital_gains.empty:
                capital_gains_df = capital_gains.reset_index()
                capital_gains_df['Symbol'] = ticker
                data['capital_gains'] = capital_gains_df
                logger.info(f"Fetched {len(capital_gains_df)} capital gains records for {ticker}")
            else:
                logger.info(f"No capital gains data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching capital gains for {ticker}: {e}")
        
        # Fetch shares full (shares outstanding data)
        try:
            shares = stock.get_shares_full()
            if shares is not None and not shares.empty:
                shares_df = shares.reset_index()
                shares_df['Symbol'] = ticker
                data['shares_full'] = shares_df
                logger.info(f"Fetched {len(shares_df)} shares outstanding records for {ticker}")
            else:
                logger.info(f"No shares outstanding data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching shares outstanding for {ticker}: {e}")
        
        # Fetch info (company information)
        try:
            info = stock.info
            if info:
                # Convert info dict to DataFrame for consistency
                info_df = pd.DataFrame([info])
                info_df['Symbol'] = ticker
                data['info'] = info_df
                logger.info(f"Fetched company info for {ticker}")
            else:
                logger.info(f"No company info available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching company info for {ticker}: {e}")
        
        # Fetch news
        try:
            news = stock.news
            if news:
                news_df = pd.DataFrame(news)
                news_df['Symbol'] = ticker
                data['news'] = news_df
                logger.info(f"Fetched {len(news_df)} news articles for {ticker}")
            else:
                logger.info(f"No news data available for {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching news for {ticker}: {e}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        return data

def save_technical_data(data: Dict[str, Any], ticker: str, output_dir: str, date_str: str) -> bool:
    """
    Save technical data to CSV files.
    
    Args:
        data (Dict[str, Any]): Technical data dictionary
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save the files
        date_str (str): Date string to include in filename
        
    Returns:
        bool: True if at least one file was saved successfully, False otherwise
    """
    try:
        # Ensure output directory exists
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        saved_files = 0
        
        # Save each type of data as a separate CSV
        data_types = ['dividends', 'splits', 'actions', 'capital_gains', 'shares_full', 'info', 'news']
        
        for data_type in data_types:
            if data[data_type] is not None and isinstance(data[data_type], pd.DataFrame) and not data[data_type].empty:
                try:
                    filename = f"{ticker}_{data_type}_{date_str}.csv"
                    filepath = os.path.join(ticker_dir, filename)
                    data[data_type].to_csv(filepath, index=False)
                    logger.info(f"Saved {data_type} data for {ticker} to {filepath}")
                    saved_files += 1
                except Exception as e:
                    logger.error(f"Error saving {data_type} data for {ticker}: {e}")
        
        # Save metadata as JSON
        try:
            metadata = {
                'ticker': ticker,
                'fetch_timestamp': data['fetch_timestamp'],
                'files_saved': saved_files,
                'data_types_available': [dt for dt in data_types if data[dt] is not None and isinstance(data[dt], pd.DataFrame) and not data[dt].empty]
            }
            metadata_filename = f"{ticker}_metadata_{date_str}.json"
            metadata_filepath = os.path.join(ticker_dir, metadata_filename)
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata for {ticker} to {metadata_filepath}")
        except Exception as e:
            logger.warning(f"Error saving metadata for {ticker}: {e}")
        
        return saved_files > 0
        
    except Exception as e:
        logger.error(f"Error saving technical data for {ticker}: {e}")
        return False

def main():
    """
    Main function to fetch and save technical data for all tickers.
    """
    # Configuration
    ticker_csv_path = "data/comprehensive_ticker_list.csv"
    output_dir = "data/technical_data"
    
    # Get today's date for filename
    today = datetime.now()
    date_str = today.strftime("%Y%m%d")
    
    logger.info(f"Starting technical data collection for {date_str}")
    
    # Load tickers
    tickers = load_tickers(ticker_csv_path)
    if not tickers:
        logger.error("No tickers loaded. Exiting.")
        return
    
    # Track statistics
    successful_downloads = 0
    failed_downloads = 0
    total_files_saved = 0
    
    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
        
        # Fetch technical data
        tech_data = fetch_technical_data(ticker)
        
        # Save data
        if save_technical_data(tech_data, ticker, output_dir, date_str):
            successful_downloads += 1
        else:
            failed_downloads += 1
            logger.warning(f"No data saved for {ticker}")
        
        # Add delay to avoid rate limiting (yfinance can be sensitive)
        time.sleep(1)  # 1 second delay between requests
        
        # Log progress every 25 tickers
        if i % 25 == 0:
            logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
    
    # Final statistics
    logger.info(f"Technical data collection completed!")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Total tickers processed: {len(tickers)}")
    logger.info(f"Success rate: {(successful_downloads/len(tickers)*100):.1f}%")

if __name__ == "__main__":
    main()