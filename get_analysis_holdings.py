import pandas as pd
import yfinance as yf
import os
import time
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_analysis_data(ticker_symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get all analysis data for a given ticker symbol.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        Dict[str, Optional[pd.DataFrame]]: Dictionary containing analysis data
    """
    ticker = yf.Ticker(ticker_symbol)
    analysis_data = {}
    
    # Analysis functions
    analysis_functions = [
        ('recommendations', lambda: ticker.recommendations),
        ('recommendations_summary', lambda: ticker.recommendations_summary),
        ('upgrades_downgrades', lambda: ticker.upgrades_downgrades),
        ('sustainability', lambda: ticker.sustainability),
        ('analyst_price_targets', lambda: ticker.analyst_price_targets),
        ('earnings_estimate', lambda: ticker.earnings_estimate),
        ('revenue_estimate', lambda: ticker.revenue_estimate),
        ('earnings_history', lambda: ticker.earnings_history),
        ('eps_trend', lambda: ticker.eps_trend),
        ('eps_revisions', lambda: ticker.eps_revisions),
        ('growth_estimates', lambda: ticker.growth_estimates)
    ]
    
    for func_name, func in analysis_functions:
        try:
            data = func()
            if data is not None and not data.empty:
                analysis_data[func_name] = data
            else:
                analysis_data[func_name] = None
                logger.warning(f"{ticker_symbol}: No data available for {func_name}")
        except Exception as e:
            logger.error(f"{ticker_symbol}: Error getting {func_name}: {str(e)}")
            analysis_data[func_name] = None
    
    return analysis_data

def get_holdings_data(ticker_symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get all holdings data for a given ticker symbol.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        Dict[str, Optional[pd.DataFrame]]: Dictionary containing holdings data
    """
    ticker = yf.Ticker(ticker_symbol)
    holdings_data = {}
    
    # Holdings functions
    holdings_functions = [
        ('funds_data', lambda: ticker.funds_data),
        ('insider_purchases', lambda: ticker.insider_purchases),
        ('insider_transactions', lambda: ticker.insider_transactions),
        ('insider_roster_holders', lambda: ticker.insider_roster_holders),
        ('major_holders', lambda: ticker.major_holders),
        ('institutional_holders', lambda: ticker.institutional_holders),
        ('mutualfund_holders', lambda: ticker.mutualfund_holders)
    ]
    
    for func_name, func in holdings_functions:
        try:
            data = func()
            if data is not None and not data.empty:
                holdings_data[func_name] = data
            else:
                holdings_data[func_name] = None
                logger.warning(f"{ticker_symbol}: No data available for {func_name}")
        except Exception as e:
            logger.error(f"{ticker_symbol}: Error getting {func_name}: {str(e)}")
            holdings_data[func_name] = None
    
    return holdings_data

def save_ticker_data(ticker_symbol: str, analysis_data: Dict[str, Optional[pd.DataFrame]], 
                    holdings_data: Dict[str, Optional[pd.DataFrame]], output_dir: str):
    """
    Save analysis and holdings data for a ticker to CSV files.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        analysis_data (Dict): Analysis data dictionary
        holdings_data (Dict): Holdings data dictionary
        output_dir (str): Output directory path
    """
    ticker_dir = os.path.join(output_dir, ticker_symbol)
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Save analysis data
    for data_type, data in analysis_data.items():
        if data is not None:
            filename = f"{ticker_symbol}_{data_type}.csv"
            filepath = os.path.join(ticker_dir, filename)
            try:
                data.to_csv(filepath, index=True)
                logger.info(f"Saved {ticker_symbol} {data_type} to {filepath}")
            except Exception as e:
                logger.error(f"Error saving {ticker_symbol} {data_type}: {str(e)}")
    
    # Save holdings data
    for data_type, data in holdings_data.items():
        if data is not None:
            filename = f"{ticker_symbol}_{data_type}.csv"
            filepath = os.path.join(ticker_dir, filename)
            try:
                data.to_csv(filepath, index=True)
                logger.info(f"Saved {ticker_symbol} {data_type} to {filepath}")
            except Exception as e:
                logger.error(f"Error saving {ticker_symbol} {data_type}: {str(e)}")

def process_all_tickers(ticker_list_path: str, output_dir: str, delay_seconds: float = 1.0):
    """
    Process all tickers from the comprehensive ticker list.
    
    Args:
        ticker_list_path (str): Path to the ticker list CSV file
        output_dir (str): Output directory for analysis data
        delay_seconds (float): Delay between API calls to avoid rate limiting
    """
    # Read ticker list
    try:
        df_tickers = pd.read_csv(ticker_list_path)
        tickers = df_tickers['Symbol'].tolist()
        logger.info(f"Found {len(tickers)} tickers to process")
    except Exception as e:
        logger.error(f"Error reading ticker list: {str(e)}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track processing statistics
    successful_tickers = []
    failed_tickers = []
    
    # Process each ticker
    total_tickers = len(tickers)
    start_time = time.time()
    
    for i, ticker_symbol in enumerate(tickers, 1):
        elapsed = time.time() - start_time
        avg_time_per_ticker = elapsed / (i - 1) if i > 1 else 0
        remaining_tickers = total_tickers - i + 1
        estimated_remaining_time = avg_time_per_ticker * remaining_tickers
        
        logger.info(f"Processing {ticker_symbol} ({i}/{total_tickers}) - "
                   f"Elapsed: {elapsed/60:.1f}m, "
                   f"ETA: {estimated_remaining_time/60:.1f}m")
        
        try:
            # Get analysis data
            analysis_data = get_analysis_data(ticker_symbol)
            
            # Get holdings data
            holdings_data = get_holdings_data(ticker_symbol)
            
            # Save data
            save_ticker_data(ticker_symbol, analysis_data, holdings_data, output_dir)
            
            successful_tickers.append(ticker_symbol)
            
            # Add delay to avoid rate limiting
            if i < total_tickers:  # Don't delay after the last ticker
                time.sleep(delay_seconds)
                
        except KeyboardInterrupt:
            logger.info(f"Process interrupted by user at ticker {ticker_symbol}")
            break
        except Exception as e:
            logger.error(f"Error processing {ticker_symbol}: {str(e)}")
            failed_tickers.append(ticker_symbol)
            continue
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"Processing completed!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Successful: {len(successful_tickers)}")
    logger.info(f"Failed: {len(failed_tickers)}")
    
    if failed_tickers:
        logger.info(f"Failed tickers: {', '.join(failed_tickers[:10])}" + 
                   (f" ... and {len(failed_tickers)-10} more" if len(failed_tickers) > 10 else ""))

def create_summary_report(output_dir: str):
    """
    Create a summary report of what data was collected for each ticker.
    
    Args:
        output_dir (str): Directory containing ticker data
    """
    summary_data = []
    
    for ticker_dir in os.listdir(output_dir):
        ticker_path = os.path.join(output_dir, ticker_dir)
        if os.path.isdir(ticker_path):
            files = os.listdir(ticker_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            analysis_files = [f for f in csv_files if any(analysis_type in f for analysis_type in [
                'recommendations', 'upgrades_downgrades', 'sustainability', 'analyst_price_targets',
                'earnings_estimate', 'revenue_estimate', 'earnings_history', 'eps_trend', 
                'eps_revisions', 'growth_estimates'
            ])]
            
            holdings_files = [f for f in csv_files if any(holdings_type in f for holdings_type in [
                'funds_data', 'insider_purchases', 'insider_transactions', 'insider_roster_holders',
                'major_holders', 'institutional_holders', 'mutualfund_holders'
            ])]
            
            summary_data.append({
                'Ticker': ticker_dir,
                'Total_Files': len(csv_files),
                'Analysis_Files': len(analysis_files),
                'Holdings_Files': len(holdings_files),
                'Files_List': ', '.join(csv_files)
            })
    
    # Create summary DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Ticker')
    summary_path = os.path.join(output_dir, 'data_collection_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary report saved to {summary_path}")
    
    # Print summary statistics
    logger.info(f"Data Collection Summary:")
    logger.info(f"Total tickers processed: {len(summary_data)}")
    logger.info(f"Average files per ticker: {summary_df['Total_Files'].mean():.2f}")
    logger.info(f"Total analysis files: {summary_df['Analysis_Files'].sum()}")
    logger.info(f"Total holdings files: {summary_df['Holdings_Files'].sum()}")

def main():
    """Main function to execute the analysis and holdings data collection."""
    # Define paths
    ticker_list_path = os.path.join('data', 'comprehensive_ticker_list.csv')
    output_dir = os.path.join('data', 'analysis_holdings')
    
    logger.info("Starting analysis and holdings data collection...")
    
    # Process all tickers
    process_all_tickers(ticker_list_path, output_dir, delay_seconds=1.0)
    
    # Create summary report
    create_summary_report(output_dir)
    
    logger.info("Analysis and holdings data collection completed!")

if __name__ == "__main__":
    main()
