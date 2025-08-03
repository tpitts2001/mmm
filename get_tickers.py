import yfinance as yf
import pandas as pd
import requests
from typing import List, Dict
import time

def get_sp500_symbols() -> List[str]:
    """Get S&P 500 stock symbols from Wikipedia"""
    try:
        # Get S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def get_nasdaq_symbols() -> List[str]:
    """Get NASDAQ stock symbols"""
    try:
        # Download NASDAQ listed companies
        url = "https://www.nasdaq.com/api/v1/screener?tableonly=true&limit=5000&offset=0"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [stock['symbol'] for stock in data.get('data', {}).get('table', {}).get('rows', [])]
    except Exception as e:
        print(f"Error fetching NASDAQ symbols: {e}")
    return []

def get_nyse_symbols() -> List[str]:
    """Get NYSE stock symbols from a reliable source"""
    try:
        # Alternative: Get from a GitHub repository that maintains stock lists
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching NYSE symbols: {e}")
        return []

def get_popular_etfs() -> List[str]:
    """Get list of popular ETFs"""
    popular_etfs = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'VEA', 'AGG', 'LQD', 'VNQ', 'GLD',
        'SLV', 'USO', 'TLT', 'HYG', 'EEM', 'FXI', 'XLF', 'XLE', 'XLK', 'XLV',
        'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC', 'VTV', 'VUG', 'VOO',
        'VXUS', 'BND', 'VTEB', 'VYM', 'SCHD', 'SCHO', 'SCHB', 'SCHX', 'SCHA',
        'SCHE', 'SCHF', 'SCHM', 'GOVT', 'VCSH', 'VCIT', 'VB', 'VO', 'VXF'
    ]
    return popular_etfs

def get_additional_large_cap_stocks() -> List[str]:
    """Get additional large cap stocks with real symbols"""
    # Expanded list of real stock symbols
    additional_stocks = [
        # Major tech companies
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'CSCO', 'UBER', 'LYFT', 'SNAP', 'TWLO',
        'SQ', 'PYPL', 'SHOP', 'ROKU', 'ZM', 'DOCU', 'OKTA', 'CRWD', 'SNOW', 'DDOG',
        'NET', 'TEAM', 'WDAY', 'VEEV', 'NOW', 'SPLK', 'FTNT', 'PANW', 'ZS', 'CYBR',
        
        # Financial sector
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
        'BLK', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO',
        'FIS', 'FISV', 'ADP', 'PAYX', 'SYF', 'DFS', 'ALLY', 'RF', 'KEY', 'FITB',
        'MTB', 'HBAN', 'CFG', 'WBS', 'ZION', 'CMA', 'PBCT', 'SIVB', 'CBOE', 'NDAQ',
        
        # Healthcare & Pharma
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'SYK', 'BSX', 'MDT', 'EW', 'DXCM',
        'ZBH', 'BAX', 'BDX', 'RMD', 'ILMN', 'IQV', 'A', 'ALGN', 'TECH', 'MRNA',
        'BNTX', 'NVAX', 'TDOC', 'VEEV', 'IDXX', 'MTD', 'DGX', 'LH', 'PKI', 'WAT',
        
        # Consumer goods & retail
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX',
        'MCD', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CL', 'KMB', 'GIS', 'K',
        'AMZN', 'EBAY', 'ETSY', 'W', 'WAYFAIR', 'CHWY', 'PETS', 'ZG', 'ZILLOW',
        'ABNB', 'DASH', 'UBER', 'LYFT', 'GPS', 'ANF', 'AEO', 'URBN', 'LULU', 'DECK',
        
        # Energy & Utilities
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'KMI', 'OKE',
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'PEG', 'SRE', 'AEP', 'PCG',
        'WMB', 'EPD', 'ET', 'MPLX', 'AM', 'PAA', 'WES', 'DCP', 'PAGP', 'TRGP',
        'ATO', 'CMS', 'DTE', 'ETR', 'ES', 'FE', 'NI', 'PNW', 'PPL', 'WEC',
        
        # Industrial & Materials
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'MMM', 'DE', 'EMR',
        'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FDX', 'CSX', 'UNP', 'NSC',
        'WM', 'RSG', 'FAST', 'PCAR', 'IR', 'OTIS', 'CARR', 'TT', 'JCI', 'SWK',
        'STZ', 'APD', 'ECL', 'FCX', 'NEM', 'AA', 'X', 'CLF', 'STLD', 'NUE',
        
        # Real Estate & REITs
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'DLR', 'WELL',
        'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'ARE', 'BXP', 'VNO', 'SLG', 'KIM',
        'REG', 'FRT', 'SPG', 'TCO', 'HST', 'HLT', 'MAR', 'IHG', 'WYNN', 'LVS',
        
        # Communication & Media
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'DISH',
        'FOXA', 'FOX', 'PARA', 'WBD', 'NWSA', 'NWS', 'NYT', 'TWTR', 'SNAP', 'PINS',
        
        # Automotive & Transportation
        'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'GRAB', 'UBER',
        'LYFT', 'DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA', 'MESA',
        
        # Biotech & Small Pharma
        'MRNA', 'BNTX', 'NVAX', 'SGEN', 'BMRN', 'RARE', 'SRPT', 'BLUE', 'FOLD', 'EDIT',
        'CRSP', 'NTLA', 'BEAM', 'VERV', 'TGTX', 'SAVA', 'BIIB', 'CELG', 'GENZ', 'HZNP',
        
        # Semiconductors
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MXIM', 'XLNX', 'LRCX',
        'AMAT', 'KLAC', 'ASML', 'TSM', 'NXPI', 'MCHP', 'SWKS', 'QRVO', 'MPWR', 'MRVL',
        
        # Software & Cloud
        'MSFT', 'ORCL', 'SAP', 'CRM', 'NOW', 'WDAY', 'ADBE', 'INTU', 'CTXS', 'VMW',
        'SPLK', 'PLTR', 'SNOW', 'DDOG', 'OKTA', 'ZS', 'CRWD', 'S', 'WORK', 'TEAM',
        
        # E-commerce & Digital
        'AMZN', 'SHOP', 'EBAY', 'ETSY', 'MELI', 'SE', 'JD', 'BABA', 'PDD', 'VIPS',
        
        # Crypto & Fintech
        'COIN', 'HOOD', 'SQ', 'PYPL', 'SOFI', 'AFRM', 'UPST', 'LC', 'MARA', 'RIOT',
        
        # Gaming & Entertainment
        'ATVI', 'EA', 'TTWO', 'RBLX', 'U', 'ZNGA', 'HUYA', 'DOYU', 'BILI', 'IQ',
        
        # Cloud Infrastructure
        'AMZN', 'MSFT', 'GOOGL', 'ORCL', 'IBM', 'VMW', 'DELL', 'HPQ', 'NTAP', 'WDC',
        
        # Food & Beverage
        'KO', 'PEP', 'MDLZ', 'GIS', 'K', 'CPB', 'CAG', 'HSY', 'SJM', 'HRL',
        'TSN', 'TYSON', 'JBS', 'BF.B', 'STZ', 'TAP', 'SAM', 'CCEP', 'KDP', 'MNST',
        
        # Additional growth stocks
        'ZOOM', 'PELOTON', 'PTON', 'BYND', 'OATLY', 'SPCE', 'OPEN', 'WISH', 'CLOV',
        'SOFI', 'PALANTIR', 'SNOWFLAKE', 'DATADOG', 'CROWDSTRIKE', 'ZSCALER', 'OKTA'
    ]
    
    # Add more symbols to reach target with more ETFs and international stocks
    # More ETFs
    additional_stocks.extend([
        'IVV', 'IEFA', 'IEMG', 'IJH', 'IJR', 'VEU', 'VTEB', 'VMOT', 'VCIT', 'VB',
        'VO', 'VTV', 'VUG', 'VYM', 'VIG', 'VGIT', 'VGSH', 'VGLT', 'VCSH', 'VCLT',
        'SPDW', 'SPEM', 'SPTM', 'SPMV', 'SPMD', 'SPYG', 'SPYV', 'SPHQ', 'SPLG',
        'DIA', 'MDY', 'SLY', 'EWJ', 'EWZ', 'EWG', 'EWU', 'EWC', 'EWH', 'EWY',
        'EWT', 'EWS', 'EWW', 'EWI', 'EWQ', 'MCHI', 'INDA', 'EPP', 'EZA', 'ECH'
    ])
    
    # International stocks (ADRs)
    additional_stocks.extend([
        'BABA', 'TSM', 'ASML', 'NVO', 'SAP', 'TM', 'SHEL', 'UL', 'SONY', 'SNY',
        'DEO', 'BTI', 'BP', 'GSK', 'AZN', 'NVS', 'ROG', 'NESN', 'RHHBY', 'LVMUY',
        'MC', 'OR', 'IDEXY', 'SMFG', 'MUFG', 'MFG', 'CNI', 'RY', 'TD', 'BNS',
        'SHOP', 'BBD', 'SU', 'CNQ', 'TRP', 'ENB', 'WCN', 'CSU', 'ATD', 'OTEX'
    ])
    
    # More US small and mid-cap stocks
    additional_stocks.extend([
        'ROKU', 'ZI', 'DOCU', 'PTON', 'PLBY', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'LC',
        'COIN', 'MARA', 'RIOT', 'HIVE', 'BITF', 'CAN', 'ARBK', 'MSTR', 'TSLA', 'NIO',
        'XPEV', 'LI', 'LCID', 'RIVN', 'GOEV', 'RIDE', 'NKLA', 'HYLN', 'WKHS', 'FSR',
        'CHPT', 'BLNK', 'EVGO', 'BLINK', 'SBE', 'QS', 'VLDR', 'LAZR', 'LIDR', 'OUST'
    ])
    
    return additional_stocks

def fetch_stock_data(symbols: List[str]) -> List[Dict]:
    """Fetch stock data for given symbols"""
    stock_data = []
    batch_size = 50  # Process in batches to avoid overwhelming the API
    
    print(f"Fetching data for {len(symbols)} symbols...")
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
        
        for symbol in batch:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get basic stock information
                stock_info = {
                    'Symbol': symbol,
                    'Name': info.get('longName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Market Cap': info.get('marketCap', 'N/A'),
                    'Current Price': info.get('currentPrice', 'N/A'),
                    'Previous Close': info.get('previousClose', 'N/A'),
                    'Volume': info.get('volume', 'N/A'),
                    'Average Volume': info.get('averageVolume', 'N/A'),
                    'PE Ratio': info.get('trailingPE', 'N/A'),
                    'Dividend Yield': info.get('dividendYield', 'N/A'),
                    'Beta': info.get('beta', 'N/A'),
                    '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                    '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                    'Exchange': info.get('exchange', 'N/A'),
                    'Currency': info.get('currency', 'N/A'),
                    'Country': info.get('country', 'N/A')
                }
                
                stock_data.append(stock_info)
                
            except Exception as e:
                # If we can't get data for a symbol, add basic info
                stock_data.append({
                    'Symbol': symbol,
                    'Name': 'Data not available',
                    'Sector': 'N/A',
                    'Industry': 'N/A',
                    'Market Cap': 'N/A',
                    'Current Price': 'N/A',
                    'Previous Close': 'N/A',
                    'Volume': 'N/A',
                    'Average Volume': 'N/A',
                    'PE Ratio': 'N/A',
                    'Dividend Yield': 'N/A',
                    'Beta': 'N/A',
                    '52 Week High': 'N/A',
                    '52 Week Low': 'N/A',
                    'Exchange': 'N/A',
                    'Currency': 'N/A',
                    'Country': 'N/A'
                })
                print(f"Could not fetch data for {symbol}: {e}")
        
        # Add a small delay to be respectful to the API
        time.sleep(1)
    
    return stock_data

def main():
    """Main function to collect 5000 top stocks and save to CSV"""
    print("Starting to collect stock symbols...")
    
    all_symbols = []
    
    # Get S&P 500 symbols
    print("Fetching S&P 500 symbols...")
    sp500_symbols = get_sp500_symbols()
    all_symbols.extend(sp500_symbols)
    print(f"Found {len(sp500_symbols)} S&P 500 symbols")
    
    # Get NASDAQ symbols
    print("Fetching NASDAQ symbols...")
    nasdaq_symbols = get_nasdaq_symbols()
    all_symbols.extend(nasdaq_symbols)
    print(f"Found {len(nasdaq_symbols)} NASDAQ symbols")
    
    # Get popular ETFs
    print("Adding popular ETFs...")
    etf_symbols = get_popular_etfs()
    all_symbols.extend(etf_symbols)
    print(f"Added {len(etf_symbols)} ETF symbols")
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    print(f"Total unique symbols so far: {len(all_symbols)}")
    
    # If we need more symbols to reach 5000, add additional ones
    if len(all_symbols) < 5000:
        print(f"Need {5000 - len(all_symbols)} more symbols...")
        additional_symbols = get_additional_large_cap_stocks()
        all_symbols.extend(additional_symbols)
        all_symbols = list(set(all_symbols))  # Remove duplicates again
    
    # Take all available symbols (don't limit to artificial 4000)
    selected_symbols = all_symbols
    print(f"Selected {len(selected_symbols)} symbols for data collection")
    
    # Fetch stock data
    stock_data = fetch_stock_data(selected_symbols)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stock_data)
    
    # Sort by market cap (descending) where available
    df['Market Cap Numeric'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    df = df.sort_values('Market Cap Numeric', ascending=False, na_position='last')
    df = df.drop('Market Cap Numeric', axis=1)  # Remove helper column
    
    # Save to CSV
    output_file = 'data/comprehensive_ticker_list.csv'
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully saved {len(df)} stocks to '{output_file}'")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()