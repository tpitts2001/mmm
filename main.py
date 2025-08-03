#!/usr/bin/env python3
"""
Main script to run the stock ticker collection process.
"""

import get_tickers
import daily_price_data

if __name__ == "__main__":
    print("Starting stock ticker collection process...")
    get_tickers.main()
    print("Ticker collection process completed!")
    
    print("Starting daily price data collection...")
    daily_price_data.main()
    print("Daily price data collection completed!")