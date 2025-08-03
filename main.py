#!/usr/bin/env python3
"""
Main script to run the stock ticker collection process.
"""

import subprocess
import sys
import os

def install_requirements():
    """
    Check if requirements.txt exists and install packages from it.
    """
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Warning: {requirements_file} not found in current directory.")
        return
    
    print("Checking and installing requirements...")
    try:
        # Install requirements using pip
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("Requirements installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print("Please install requirements manually using: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

import get_tickers
import daily_price_data

if __name__ == "__main__":
    # Install requirements first
    install_requirements()
    
    print("Starting stock ticker collection process...")
    get_tickers.main()
    print("Ticker collection process completed!")
    
    print("Starting daily price data collection...")
    daily_price_data.main()
    print("Daily price data collection completed!")