#!/usr/bin/env python3
"""
Test script to run the LSTM model with synthetic data to verify functionality
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model_lstm import StockPredictor, load_and_prepare_stock_data
import matplotlib.pyplot as plt

def create_synthetic_stock_data(days=500, seed=42):
    """
    Create synthetic stock data for testing the LSTM model
    """
    np.random.seed(seed)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate synthetic price data with trend and volatility
    initial_price = 100
    price_changes = np.random.normal(0.001, 0.02, days)  # Small daily returns with volatility
    
    # Add some trend and cycles
    trend = np.linspace(0, 0.5, days)  # Gradual upward trend
    cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, days))  # Cyclic pattern
    
    prices = [initial_price]
    for i in range(1, days):
        price_change = price_changes[i] + trend[i]/days + cycle[i]/days
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create OHLCV data
    close_prices = np.array(prices)
    
    # Generate realistic OHLC from close prices
    daily_volatility = 0.01
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility, days))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility, days))
    
    # Open prices (close of previous day with small gap)
    open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, 0.005, days))
    open_prices[0] = close_prices[0]
    
    # Ensure OHLC relationships are maintained
    for i in range(days):
        high_prices[i] = max(open_prices[i], high_prices[i], low_prices[i], close_prices[i])
        low_prices[i] = min(open_prices[i], high_prices[i], low_prices[i], close_prices[i])
    
    # Generate volume data
    volume = np.random.randint(100000, 1000000, days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    return df

def test_lstm_model():
    """
    Test the LSTM model with synthetic data
    """
    print("Creating synthetic stock data...")
    df = create_synthetic_stock_data(days=300)
    
    print("Data shape:", df.shape)
    print("Data sample:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    
    # Initialize predictor
    print("\nInitializing StockPredictor...")
    predictor = StockPredictor(sequence_length=20)
    
    try:
        # Prepare data
        print("Preparing data with technical indicators...")
        features, targets = predictor.prepare_data(df, target_column='Close')
        
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = predictor.create_data_loaders(features, targets, batch_size=16)
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Train model (with fewer epochs for testing)
        print("Training model...")
        train_losses, val_losses = predictor.train_model(
            train_loader, val_loader, 
            input_size=features.shape[1], 
            epochs=20,  # Reduced for testing
            learning_rate=0.001
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        mse, mae = predictor.evaluate_model(features, targets)
        
        # Plot training progress
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Make predictions for visualization
        predictions = predictor.predict(features)
        actual_targets = predictor.scaler_targets.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # Plot predictions vs actual (last 50 points for clarity)
        plt.subplot(1, 2, 2)
        plot_range = slice(-50, None)
        plt.plot(actual_targets[predictor.sequence_length:][plot_range], label='Actual', alpha=0.7)
        plt.plot(predictions[:len(actual_targets[predictor.sequence_length:])][plot_range], label='Predicted', alpha=0.7)
        plt.title('Predictions vs Actual (Last 50 points)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('lstm_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nTest completed successfully!")
        print(f"Final MSE: {mse:.6f}")
        print(f"Final MAE: {mae:.6f}")
        print("Results saved to 'lstm_test_results.png'")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing LSTM Stock Prediction Model...")
    print("=" * 50)
    
    success = test_lstm_model()
    
    if success:
        print("\n" + "=" * 50)
        print("✓ LSTM Model test passed!")
        print("The model is working correctly with synthetic data.")
    else:
        print("\n" + "=" * 50)
        print("✗ LSTM Model test failed!")
        print("Check the error messages above for debugging information.")
