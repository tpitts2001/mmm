import os
import pandas as pd
from typing import Dict, List

def analyze_financial_data_collection(data_dir: str = r"data\financial_data") -> Dict:
    """
    Analyze the collected financial data and provide a summary.
    
    Args:
        data_dir: Directory containing the financial data
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return {}
    
    ticker_folders = [name for name in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, name))]
    
    analysis = {
        'total_tickers_processed': len(ticker_folders),
        'tickers_with_data': [],
        'tickers_without_data': [],
        'file_type_counts': {},
        'detailed_results': {}
    }
    
    expected_files = [
        'income_statement.csv',
        'quarterly_income_statement.csv', 
        'ttm_income_statement.csv',
        'balance_sheet.csv',
        'quarterly_balance_sheet.csv',
        'cash_flow.csv',
        'quarterly_cash_flow.csv',
        'ttm_cash_flow.csv',
        'earnings_dates.csv',
        'calendar.csv',
        'earnings.csv',
        'sec_filings.csv'
    ]
    
    for file_type in expected_files:
        analysis['file_type_counts'][file_type] = 0
    
    for ticker in sorted(ticker_folders):
        ticker_path = os.path.join(data_dir, ticker)
        files = os.listdir(ticker_path)
        
        if files:
            analysis['tickers_with_data'].append(ticker)
            analysis['detailed_results'][ticker] = {
                'file_count': len(files),
                'files': files
            }
            
            # Count file types
            for file in files:
                for expected_file in expected_files:
                    if file.endswith(expected_file):
                        analysis['file_type_counts'][expected_file] += 1
                        break
        else:
            analysis['tickers_without_data'].append(ticker)
            analysis['detailed_results'][ticker] = {
                'file_count': 0,
                'files': []
            }
    
    return analysis

def print_summary(analysis: Dict):
    """Print a formatted summary of the analysis."""
    print("=" * 60)
    print("FINANCIAL DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal tickers processed: {analysis['total_tickers_processed']}")
    print(f"Tickers with data: {len(analysis['tickers_with_data'])}")
    print(f"Tickers without data: {len(analysis['tickers_without_data'])}")
    
    print(f"\nData collection success rate: {len(analysis['tickers_with_data'])/analysis['total_tickers_processed']*100:.1f}%")
    
    print("\n" + "-" * 40)
    print("FILE TYPE AVAILABILITY")
    print("-" * 40)
    
    for file_type, count in analysis['file_type_counts'].items():
        if count > 0:
            percentage = count / len(analysis['tickers_with_data']) * 100 if analysis['tickers_with_data'] else 0
            print(f"{file_type:<35}: {count:>3} tickers ({percentage:>5.1f}%)")
    
    if analysis['tickers_without_data']:
        print(f"\n" + "-" * 40)
        print("TICKERS WITHOUT DATA")
        print("-" * 40)
        for ticker in analysis['tickers_without_data'][:10]:  # Show first 10
            print(f"  {ticker}")
        if len(analysis['tickers_without_data']) > 10:
            print(f"  ... and {len(analysis['tickers_without_data']) - 10} more")
    
    print(f"\n" + "-" * 40)
    print("SAMPLE OF SUCCESSFUL TICKERS")
    print("-" * 40)
    sample_tickers = list(analysis['tickers_with_data'])[:5]
    for ticker in sample_tickers:
        files = analysis['detailed_results'][ticker]['files']
        print(f"{ticker}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"    {file}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more files")

def export_summary_csv(analysis: Dict, output_file: str = "financial_data_summary.csv"):
    """Export the summary to a CSV file."""
    summary_data = []
    
    for ticker, details in analysis['detailed_results'].items():
        summary_data.append({
            'Ticker': ticker,
            'Files_Count': details['file_count'],
            'Has_Data': 'Yes' if details['file_count'] > 0 else 'No',
            'Files': '; '.join(details['files'])
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed summary exported to: {output_file}")

def main():
    """Main function to run the analysis."""
    analysis = analyze_financial_data_collection()
    
    if analysis:
        print_summary(analysis)
        export_summary_csv(analysis)
    else:
        print("No data found to analyze.")

if __name__ == "__main__":
    main()
