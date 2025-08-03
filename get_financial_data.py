import os
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """
    A class to collect comprehensive financial data for stocks using yfinance.
    Saves data as CSV files organized by ticker symbol.
    """
    
    def __init__(self, ticker_csv_path: str, output_dir: str):
        """
        Initialize the FinancialDataCollector.
        
        Args:
            ticker_csv_path: Path to CSV file containing ticker symbols
            output_dir: Directory to save the financial data CSV files
        """
        self.ticker_csv_path = ticker_csv_path
        self.output_dir = output_dir
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def load_ticker_symbols(self) -> List[str]:
        """
        Load ticker symbols from the CSV file.
        
        Returns:
            List of ticker symbols
        """
        try:
            df = pd.read_csv(self.ticker_csv_path)
            if 'Symbol' not in df.columns:
                raise ValueError("CSV file must contain a 'Symbol' column")
            
            tickers = df['Symbol'].dropna().unique().tolist()
            logger.info(f"Loaded {len(tickers)} ticker symbols")
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading ticker symbols: {e}")
            raise
    
    def get_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive financial data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing all financial data
        """
        try:
            stock = yf.Ticker(ticker)
            
            financial_data = {
                'ticker': ticker,
                'income_statement': None,
                'quarterly_income_statement': None,
                'ttm_income_statement': None,
                'balance_sheet': None,
                'quarterly_balance_sheet': None,
                'cash_flow': None,
                'quarterly_cash_flow': None,
                'ttm_cash_flow': None,
                'calendar': None,
                'earnings_dates': None,
                'sec_filings': None
            }
            
            # Get Income Statement (Annual)
            try:
                financial_data['income_statement'] = stock.financials
                logger.debug(f"{ticker}: Retrieved annual income statement")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get annual income statement - {e}")
            
            # Get Quarterly Income Statement
            try:
                financial_data['quarterly_income_statement'] = stock.quarterly_financials
                logger.debug(f"{ticker}: Retrieved quarterly income statement")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get quarterly income statement - {e}")
            
            # Get TTM Income Statement (using latest quarterly data)
            try:
                quarterly_data = stock.quarterly_financials
                if quarterly_data is not None and not quarterly_data.empty:
                    # Take the last 4 quarters for TTM calculation
                    financial_data['ttm_income_statement'] = quarterly_data.iloc[:, :4].sum(axis=1).to_frame('TTM')
                logger.debug(f"{ticker}: Calculated TTM income statement")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to calculate TTM income statement - {e}")
            
            # Get Balance Sheet (Annual)
            try:
                financial_data['balance_sheet'] = stock.balance_sheet
                logger.debug(f"{ticker}: Retrieved annual balance sheet")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get annual balance sheet - {e}")
            
            # Get Quarterly Balance Sheet
            try:
                financial_data['quarterly_balance_sheet'] = stock.quarterly_balance_sheet
                logger.debug(f"{ticker}: Retrieved quarterly balance sheet")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get quarterly balance sheet - {e}")
            
            # Get Cash Flow (Annual)
            try:
                financial_data['cash_flow'] = stock.cashflow
                logger.debug(f"{ticker}: Retrieved annual cash flow")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get annual cash flow - {e}")
            
            # Get Quarterly Cash Flow
            try:
                financial_data['quarterly_cash_flow'] = stock.quarterly_cashflow
                logger.debug(f"{ticker}: Retrieved quarterly cash flow")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get quarterly cash flow - {e}")
            
            # Get TTM Cash Flow
            try:
                quarterly_cf = stock.quarterly_cashflow
                if quarterly_cf is not None and not quarterly_cf.empty:
                    # Take the last 4 quarters for TTM calculation
                    financial_data['ttm_cash_flow'] = quarterly_cf.iloc[:, :4].sum(axis=1).to_frame('TTM')
                logger.debug(f"{ticker}: Calculated TTM cash flow")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to calculate TTM cash flow - {e}")
            
            # Note: stock.earnings is deprecated - Net Income can be found in income_stmt instead
            
            # Get Calendar
            try:
                financial_data['calendar'] = stock.calendar
                logger.debug(f"{ticker}: Retrieved calendar")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get calendar - {e}")
            
            # Get Earnings Dates
            try:
                financial_data['earnings_dates'] = stock.earnings_dates
                logger.debug(f"{ticker}: Retrieved earnings dates")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get earnings dates - {e}")
            
            # Get SEC Filings
            try:
                financial_data['sec_filings'] = stock.sec_filings
                logger.debug(f"{ticker}: Retrieved SEC filings")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get SEC filings - {e}")
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error getting financial data for {ticker}: {e}")
            return None
    
    def save_ticker_data(self, ticker: str, financial_data: Dict[str, Any]):
        """
        Save financial data for a ticker to CSV files.
        
        Args:
            ticker: Stock ticker symbol
            financial_data: Dictionary containing financial data
        """
        ticker_dir = os.path.join(self.output_dir, ticker)
        if not os.path.exists(ticker_dir):
            os.makedirs(ticker_dir)
        
        saved_files = []
        
        for data_type, data in financial_data.items():
            if data_type == 'ticker' or data is None:
                continue
                
            try:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filename = f"{ticker}_{data_type}.csv"
                    filepath = os.path.join(ticker_dir, filename)
                    data.to_csv(filepath)
                    saved_files.append(filename)
                    logger.debug(f"Saved {filename}")
                elif isinstance(data, pd.Series) and not data.empty:
                    filename = f"{ticker}_{data_type}.csv"
                    filepath = os.path.join(ticker_dir, filename)
                    data.to_csv(filepath)
                    saved_files.append(filename)
                    logger.debug(f"Saved {filename}")
                else:
                    logger.debug(f"{ticker}: No data available for {data_type}")
                    
            except Exception as e:
                logger.warning(f"Failed to save {data_type} for {ticker}: {e}")
        
        if saved_files:
            logger.info(f"{ticker}: Saved {len(saved_files)} files - {', '.join(saved_files)}")
        else:
            logger.warning(f"{ticker}: No files were saved")
    
    def collect_all_data(self, delay_seconds: float = 0.1, skip_existing: bool = True):
        """
        Collect financial data for all tickers and save to CSV files.
        
        Args:
            delay_seconds: Delay between API calls to respect rate limits
            skip_existing: Skip tickers that already have data folders
        """
        tickers = self.load_ticker_symbols()
        
        # Filter out already processed tickers if skip_existing is True
        if skip_existing:
            existing_tickers = set()
            if os.path.exists(self.output_dir):
                existing_tickers = {name for name in os.listdir(self.output_dir) 
                                  if os.path.isdir(os.path.join(self.output_dir, name))}
            
            original_count = len(tickers)
            tickers = [ticker for ticker in tickers if ticker not in existing_tickers]
            
            if existing_tickers:
                logger.info(f"Skipping {len(existing_tickers)} already processed tickers")
                logger.info(f"Remaining tickers to process: {len(tickers)} out of {original_count}")
        
        total_tickers = len(tickers)
        
        logger.info(f"Starting data collection for {total_tickers} tickers")
        
        if total_tickers == 0:
            logger.info("No tickers to process. All tickers have already been processed.")
            return
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{total_tickers})")
            
            try:
                financial_data = self.get_financial_data(ticker)
                
                if financial_data:
                    self.save_ticker_data(ticker, financial_data)
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    logger.warning(f"Failed to get data for {ticker}")
                
                # Add delay to respect API rate limits
                if i < total_tickers:
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                failed_downloads += 1
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        logger.info(f"Data collection complete!")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Data saved to: {self.output_dir}")


def main():
    """Main function to run the financial data collection."""
    # Define paths
    ticker_csv_path = r"data\comprehensive_ticker_list.csv"
    output_dir = r"data\financial_data"
    
    # Create collector instance
    collector = FinancialDataCollector(ticker_csv_path, output_dir)
    
    # Collect all financial data
    collector.collect_all_data(delay_seconds=0.2)


if __name__ == "__main__":
    main()
