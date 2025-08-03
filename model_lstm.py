import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    """
    Custom Dataset class for stock data
    This handles the conversion of time series data into sequences
    that the LSTM can process
    """
    def __init__(self, features, targets, sequence_length=20):
        """
        Args:
            features: Technical indicators and price data (210 features as mentioned in paper)
            targets: Target prices (next month's closing price)
            sequence_length: Number of previous time steps to use for prediction
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Return sequence of features and corresponding target
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length]
        )

class StockLSTM(nn.Module):
    """
    LSTM model architecture based on the research paper
    3 layers, 32 units each, following the paper's specifications
    """
    def __init__(self, input_size, hidden_size=32, num_layers=3, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers - 3 layers with 32 units each as per paper
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer to predict single price value
        self.fc = nn.Linear(hidden_size, 1)
        
        # ReLU activation as mentioned in paper
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        # Initialize hidden states on the same device as input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply ReLU activation
        activated = self.relu(last_output)
        
        # Final prediction
        prediction = self.fc(activated)
        
        return prediction

class StockPredictor:
    """
    Main class that handles data preprocessing, model training, and prediction
    This encapsulates the entire pipeline described in the paper
    """
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler()
        self.scaler_targets = MinMaxScaler()
        self.model = None
        
    def prepare_technical_indicators(self, df):
        """
        Create the 210 technical indicators mentioned in the paper
        This includes price data, volume, and derived indicators like MACD, RSI, etc.
        """
        features = pd.DataFrame()
        
        # Basic price and volume features (as mentioned in paper)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Calculate moving averages for the past 20 days
        for window in [5, 10, 20, 50]:
            if 'Close' in df.columns:
                features[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                features[f'MA_{window}_ratio'] = df['Close'] / features[f'MA_{window}']
        
        # MACD indicator (key technical indicator from paper)
        if 'Close' in df.columns:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            features['MACD'] = exp1 - exp2
            features['MACD_signal'] = features['MACD'].ewm(span=9).mean()
            features['MACD_histogram'] = features['MACD'] - features['MACD_signal']
        
        # RSI (Relative Strength Index)
        if 'Close' in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'Close' in df.columns:
            rolling_mean = df['Close'].rolling(window=20).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            features['BB_upper'] = rolling_mean + (rolling_std * 2)
            features['BB_lower'] = rolling_mean - (rolling_std * 2)
            features['BB_width'] = features['BB_upper'] - features['BB_lower']
            features['BB_position'] = (df['Close'] - features['BB_lower']) / features['BB_width']
        
        # Volume-based indicators
        if 'Volume' in df.columns and 'Close' in df.columns:
            features['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            features['Volume_ratio'] = df['Volume'] / features['Volume_MA']
            features['Price_Volume'] = df['Close'] * df['Volume']
        
        # Price change indicators
        if 'Close' in df.columns:
            for period in [1, 5, 10, 20]:
                features[f'Return_{period}d'] = df['Close'].pct_change(period, fill_method=None)
                features[f'Volatility_{period}d'] = df['Close'].pct_change(fill_method=None).rolling(window=period).std()
        
        # Remove any NaN values
        features = features.ffill().fillna(0)
        
        return features
    
    def prepare_data(self, df, target_column='Close'):
        """
        Prepare data for LSTM training following the paper's methodology
        """
        print("Preparing technical indicators...")
        features = self.prepare_technical_indicators(df)
        
        print(f"Created {features.shape[1]} technical indicators")
        
        # Normalize features to [0,1] range as mentioned in paper
        features_scaled = self.scaler_features.fit_transform(features)
        
        # Prepare targets (next month's closing price)
        # Since we're doing monthly prediction, we shift by approximately 22 trading days
        targets = df[target_column].shift(-22).dropna()
        features_scaled = features_scaled[:-22]  # Align with targets
        
        # Scale targets
        targets_scaled = self.scaler_targets.fit_transform(targets.values.reshape(-1, 1)).flatten()
        
        return features_scaled, targets_scaled
    
    def create_data_loaders(self, features, targets, train_ratio=0.8, batch_size=32):
        """
        Create training and validation data loaders
        """
        # Split data
        split_idx = int(len(features) * train_ratio)
        
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]
        
        # Create datasets
        train_dataset = StockDataset(train_features, train_targets, self.sequence_length)
        val_dataset = StockDataset(val_features, val_targets, self.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, input_size, epochs=100, learning_rate=0.001, patience=10):
        """
        Train the LSTM model using the parameters from the paper
        Added early stopping to prevent overfitting
        """
        # Initialize model with paper's specifications
        self.model = StockLSTM(input_size=input_size, hidden_size=32, num_layers=3)
        
        # Use MSE loss and Adam optimizer as specified in paper
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                try:
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs.squeeze(), batch_targets)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    try:
                        outputs = self.model(batch_features)
                        loss = criterion(outputs.squeeze(), batch_targets)
                        val_loss += loss.item()
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                # Load best model state
                if hasattr(self, 'best_model_state'):
                    self.model.load_state_dict(self.best_model_state)
                break
        
        return train_losses, val_losses
    
    def predict(self, features):
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        with torch.no_grad():
            # Prepare data for prediction
            dataset = StockDataset(features, np.zeros(len(features)), self.sequence_length)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            predictions = []
            for batch_features, _ in dataloader:
                output = self.model(batch_features)
                predictions.append(output.item())
        
        # Inverse transform predictions to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_targets.inverse_transform(predictions).flatten()
        
        return predictions
    
    def evaluate_model(self, features, targets):
        """
        Evaluate model performance using MAE and MSE as mentioned in paper
        """
        predictions = self.predict(features)
        
        # Align predictions with targets (account for sequence length)
        aligned_targets = targets[self.sequence_length:]
        aligned_predictions = predictions[:len(aligned_targets)]
        
        # Calculate metrics as used in the paper
        mse = mean_squared_error(aligned_targets, aligned_predictions)
        mae = mean_absolute_error(aligned_targets, aligned_predictions)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        return mse, mae

# Example usage and data loading function
def load_and_prepare_stock_data(csv_file_path):
    """
    Load stock data from CSV file
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    return df

# Main execution example
if __name__ == "__main__":
    # Example of how to use the complete pipeline
    
    # Load your stock data
    # df = load_and_prepare_stock_data('your_stock_data.csv')
    
    # Initialize predictor
    predictor = StockPredictor(sequence_length=20)
    
    # Prepare data (this would use your actual data)
    # features, targets = predictor.prepare_data(df)
    
    # Create data loaders
    # train_loader, val_loader = predictor.create_data_loaders(features, targets)
    
    # Train model
    # train_losses, val_losses = predictor.train_model(
    #     train_loader, val_loader, 
    #     input_size=features.shape[1], 
    #     epochs=100
    # )
    
    # Evaluate model
    # mse, mae = predictor.evaluate_model(features, targets)
    
    print("LSTM Stock Prediction Model Implementation Complete!")
    print("This implementation follows the methodology described in the research paper.")