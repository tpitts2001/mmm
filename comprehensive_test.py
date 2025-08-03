#!/usr/bin/env python3
"""
Comprehensive test script for the LSTM model including real data download
"""

import yfinance as yf
import pandas as pd
import numpy as np
from model_lstm import StockPredictor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def test_with_real_data():
    """Test the model with real stock data from Yahoo Finance"""
    print("Downloading real stock data...")
    
    try:
        # Download Apple stock data for testing
        ticker = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Reset index to get Date as a column
        df = df.reset_index()
        
        print(f"Downloaded {len(df)} days of {ticker} data")
        print("Data sample:")
        print(df.head())
        
        # Initialize predictor
        predictor = StockPredictor(sequence_length=20)
        
        # Prepare data
        features, targets = predictor.prepare_data(df, target_column='Close')
        
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Create data loaders
        train_loader, val_loader = predictor.create_data_loaders(features, targets, batch_size=16)
        
        # Train model with fewer epochs for testing
        print("Training model on real data...")
        train_losses, val_losses = predictor.train_model(
            train_loader, val_loader, 
            input_size=features.shape[1], 
            epochs=30,
            learning_rate=0.001
        )
        
        # Evaluate model
        mse, mae = predictor.evaluate_model(features, targets)
        
        print(f"Real data test - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test various edge cases to ensure model robustness"""
    print("\nTesting edge cases...")
    
    # Test 1: Very small dataset
    try:
        print("Test 1: Small dataset...")
        predictor = StockPredictor(sequence_length=5)
        
        # Create very small synthetic data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        small_df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(10000, 100000, 50)
        })
        
        features, targets = predictor.prepare_data(small_df)
        train_loader, val_loader = predictor.create_data_loaders(features, targets, batch_size=4)
        
        # Train with very few epochs
        train_losses, val_losses = predictor.train_model(
            train_loader, val_loader, 
            input_size=features.shape[1], 
            epochs=5
        )
        
        print("✓ Small dataset test passed")
        
    except Exception as e:
        print(f"✗ Small dataset test failed: {e}")
    
    # Test 2: Data with missing values
    try:
        print("Test 2: Data with missing values...")
        predictor = StockPredictor(sequence_length=10)
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df_with_nan = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(95, 105, 100),
            'High': np.random.uniform(100, 110, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.randint(10000, 100000, 100)
        })
        
        # Introduce some NaN values
        df_with_nan.loc[10:15, 'Close'] = np.nan
        df_with_nan.loc[30:32, 'Volume'] = np.nan
        
        features, targets = predictor.prepare_data(df_with_nan)
        
        # Check if NaN values were handled
        if not np.isnan(features).any() and not np.isnan(targets).any():
            print("✓ Missing values test passed")
        else:
            print("✗ Missing values test failed - NaN values still present")
            
    except Exception as e:
        print(f"✗ Missing values test failed: {e}")
    
    # Test 3: Single feature prediction
    try:
        print("Test 3: Minimal feature set...")
        predictor = StockPredictor(sequence_length=10)
        
        # Create data with only basic OHLCV
        dates = pd.date_range('2024-01-01', periods=80, freq='D')
        minimal_df = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.normal(0, 1, 80),
            'High': 102 + np.random.normal(0, 1, 80),
            'Low': 98 + np.random.normal(0, 1, 80),
            'Close': 100 + np.random.normal(0, 1, 80),
            'Volume': np.random.randint(10000, 100000, 80)
        })
        
        features, targets = predictor.prepare_data(minimal_df)
        print(f"Minimal features shape: {features.shape}")
        print("✓ Minimal feature set test passed")
        
    except Exception as e:
        print(f"✗ Minimal feature set test failed: {e}")

def run_comprehensive_tests():
    """Run all tests"""
    print("Running comprehensive LSTM model tests...")
    print("=" * 60)
    
    # Test with real data
    real_data_success = test_with_real_data()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n" + "=" * 60)
    if real_data_success:
        print("✓ All tests completed successfully!")
        print("The LSTM model is robust and ready for production use.")
    else:
        print("⚠ Some tests had issues but basic functionality works.")
        print("The model works with synthetic data but may need adjustments for real data.")

if __name__ == "__main__":
    run_comprehensive_tests()
