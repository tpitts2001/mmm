import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AAPLDataProcessor:
    def __init__(self):
        self.price_scaler = StandardScaler()
        self.fundamental_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        self.tokenizer = None
        self.sentiment_model = None
        
    def load_and_process_data(self, file_paths, start_date='2020-01-01', end_date='2025-08-01'):
        """Load and align all data sources into a unified dataset"""
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        master_df = pd.DataFrame({'date': date_range})
        master_df.set_index('date', inplace=True)
        
        # Process price data (you'll need to load your historical price data)
        # For demonstration, creating synthetic price data structure
        price_data = self._create_sample_price_data(date_range)
        master_df = master_df.join(price_data, how='left')
        
        # Process fundamental data
        fundamental_data = self._process_fundamental_data(file_paths)
        master_df = master_df.join(fundamental_data, how='left')
        
        # Process news sentiment
        news_sentiment = self._process_news_data(file_paths)
        master_df = master_df.join(news_sentiment, how='left')
        
        # Process corporate actions
        corporate_actions = self._process_corporate_actions(file_paths)
        master_df = master_df.join(corporate_actions, how='left')
        
        # Forward fill and clean data
        master_df = self._clean_and_fill_data(master_df)
        
        return master_df
    
    def _create_sample_price_data(self, date_range):
        """Create sample price data structure - replace with your actual price data loading"""
        np.random.seed(42)
        prices = 150 + np.cumsum(np.random.randn(len(date_range)) * 2)
        volumes = np.random.exponential(50000000, len(date_range))
        
        price_df = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, len(date_range)),
            'high': prices * np.random.uniform(1.00, 1.05, len(date_range)),
            'low': prices * np.random.uniform(0.95, 1.00, len(date_range)),
            'close': prices,
            'volume': volumes,
            'returns': np.concatenate([[0], np.diff(np.log(prices))])
        }, index=date_range)
        
        return price_df
    
    def _process_fundamental_data(self, file_paths):
        """Process quarterly financial statements"""
        fundamental_features = []
        
        # Key fundamental metrics to extract
        key_metrics = [
            'total_revenue', 'gross_profit', 'operating_income', 'net_income',
            'total_assets', 'total_debt', 'shareholders_equity', 'free_cash_flow',
            'current_assets', 'current_liabilities'
        ]
        
        # Create quarterly date index (sample dates)
        quarterly_dates = pd.date_range('2020-01-01', '2025-08-01', freq='Q')
        
        # Generate sample fundamental data - replace with actual file loading
        fundamental_data = {}
        for metric in key_metrics:
            # Create trending fundamental data
            base_values = np.random.exponential(1e9, len(quarterly_dates))
            growth_trend = 1.02 ** np.arange(len(quarterly_dates))  # 2% quarterly growth
            fundamental_data[f'fundamental_{metric}'] = base_values * growth_trend
        
        fundamental_df = pd.DataFrame(fundamental_data, index=quarterly_dates)
        
        # Calculate derived ratios
        fundamental_df['roe'] = fundamental_df['fundamental_net_income'] / fundamental_df['fundamental_shareholders_equity']
        fundamental_df['debt_to_equity'] = fundamental_df['fundamental_total_debt'] / fundamental_df['fundamental_shareholders_equity']
        fundamental_df['current_ratio'] = fundamental_df['fundamental_current_assets'] / fundamental_df['fundamental_current_liabilities']
        
        # Resample to daily frequency with forward fill
        daily_fundamentals = fundamental_df.resample('D').ffill()
        
        return daily_fundamentals
    
    def _process_news_data(self, file_paths):
        """Process news sentiment data"""
        # Create sample news sentiment data
        date_range = pd.date_range('2020-01-01', '2025-08-01', freq='D')
        
        # Simulate sentiment scores
        np.random.seed(42)
        sentiment_data = pd.DataFrame({
            'sentiment_score': np.random.normal(0.1, 0.3, len(date_range)),  # Slightly positive bias
            'sentiment_magnitude': np.random.exponential(0.5, len(date_range)),
            'news_count': np.random.poisson(3, len(date_range)),
            'positive_mentions': np.random.poisson(2, len(date_range)),
            'negative_mentions': np.random.poisson(1, len(date_range))
        }, index=date_range)
        
        return sentiment_data
    
    def _process_corporate_actions(self, file_paths):
        """Process corporate actions and events"""
        date_range = pd.date_range('2020-01-01', '2025-08-01', freq='D')
        
        # Create binary indicators for events
        actions_data = pd.DataFrame({
            'earnings_announcement': np.zeros(len(date_range)),
            'dividend_ex_date': np.zeros(len(date_range)),
            'stock_split': np.zeros(len(date_range)),
            'insider_trading': np.random.binomial(1, 0.02, len(date_range))  # 2% chance per day
        }, index=date_range)
        
        # Mark quarterly earnings announcements
        earnings_dates = pd.date_range('2020-01-15', '2025-08-01', freq='Q')
        for date in earnings_dates:
            if date in actions_data.index:
                actions_data.loc[date, 'earnings_announcement'] = 1
        
        # Mark quarterly dividend dates
        dividend_dates = pd.date_range('2020-02-07', '2025-08-01', freq='Q')
        for date in dividend_dates:
            if date in actions_data.index:
                actions_data.loc[date, 'dividend_ex_date'] = 1
        
        return actions_data
    
    def _clean_and_fill_data(self, df):
        """Clean and prepare final dataset"""
        # Forward fill fundamental data
        fundamental_cols = [col for col in df.columns if 'fundamental' in col or col in ['roe', 'debt_to_equity', 'current_ratio']]
        df[fundamental_cols] = df[fundamental_cols].ffill()
        
        # Fill sentiment data (use 0 for missing sentiment)
        sentiment_cols = [col for col in df.columns if 'sentiment' in col or 'news' in col or 'mentions' in col]
        df[sentiment_cols] = df[sentiment_cols].fillna(0)
        
        # Remove weekends and holidays (basic business day filter)
        df = df[df.index.dayofweek < 5]
        
        # Drop rows with insufficient data
        df = df.dropna()
        
        return df

class AAPLDataset(Dataset):
    def __init__(self, data, sequence_length=60, prediction_horizon=5):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Separate feature types
        self.price_features = ['open', 'high', 'low', 'close', 'volume', 'returns']
        self.fundamental_features = [col for col in data.columns if 'fundamental' in col or col in ['roe', 'debt_to_equity', 'current_ratio']]
        self.sentiment_features = [col for col in data.columns if 'sentiment' in col or 'news' in col or 'mentions' in col]
        self.action_features = ['earnings_announcement', 'dividend_ex_date', 'stock_split', 'insider_trading']
        
        self.price_data = data[self.price_features].values
        self.fundamental_data = data[self.fundamental_features].values
        self.sentiment_data = data[self.sentiment_features].values
        self.action_data = data[self.action_features].values
        
        # Calculate future returns as targets
        self.targets = self._calculate_targets(data['close'].values)
        
    def _calculate_targets(self, prices):
        """Calculate future returns at different horizons"""
        targets = []
        for i in range(len(prices) - self.prediction_horizon):
            future_return = (prices[i + self.prediction_horizon] - prices[i]) / prices[i]
            targets.append([
                future_return,  # Raw return
                1 if future_return > 0 else 0  # Direction (classification)
            ])
        
        return np.array(targets)
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon
    
    def __getitem__(self, idx):
        # Get sequences
        price_seq = self.price_data[idx:idx + self.sequence_length]
        fundamental_seq = self.fundamental_data[idx:idx + self.sequence_length]
        sentiment_seq = self.sentiment_data[idx:idx + self.sequence_length]
        action_seq = self.action_data[idx:idx + self.sequence_length]
        
        target = self.targets[idx + self.sequence_length - 1]
        
        return {
            'price': torch.FloatTensor(price_seq),
            'fundamental': torch.FloatTensor(fundamental_seq),
            'sentiment': torch.FloatTensor(sentiment_seq),
            'actions': torch.FloatTensor(action_seq),
            'target': torch.FloatTensor(target)
        }

class MultiModalStockPredictor(nn.Module):
    def __init__(self, price_features, fundamental_features, sentiment_features, action_features, 
                 hidden_dim=128, num_layers=2, dropout=0.2):
        super(MultiModalStockPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Price stream - LSTM for temporal patterns
        self.price_lstm = nn.LSTM(
            price_features, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Fundamental stream - Dense layers
        self.fundamental_encoder = nn.Sequential(
            nn.Linear(fundamental_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Sentiment stream - CNN for local patterns
        self.sentiment_conv = nn.Sequential(
            nn.Conv1d(sentiment_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Corporate actions - Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(action_features, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Multi-head attention for fusion
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=dropout)
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [return_prediction, direction_probability]
        )
        
        self.return_head = nn.Linear(2, 1)  # Regression head
        self.direction_head = nn.Sequential(  # Classification head
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, price, fundamental, sentiment, actions):
        batch_size, seq_len = price.shape[0], price.shape[1]
        
        # Process price sequence
        price_out, (h_n, c_n) = self.price_lstm(price)
        price_features = price_out[:, -1, :]  # Take last timestep
        
        # Process fundamental data (use last timestep)
        fundamental_features = self.fundamental_encoder(fundamental[:, -1, :])
        
        # Process sentiment with CNN
        sentiment_transposed = sentiment.transpose(1, 2)  # (batch, features, seq_len)
        sentiment_conv_out = self.sentiment_conv(sentiment_transposed)
        sentiment_features = sentiment_conv_out.squeeze(-1)
        
        # Process corporate actions (average over sequence)
        action_features = self.action_embedding(actions.mean(dim=1))
        
        # Concatenate all features
        combined_features = torch.cat([
            price_features, fundamental_features, sentiment_features, action_features
        ], dim=1)
        
        # Apply self-attention (treating as single sequence element)
        combined_features = combined_features.unsqueeze(0)  # (1, batch, features)
        attn_out, _ = self.attention(combined_features, combined_features, combined_features)
        attn_features = attn_out.squeeze(0)  # (batch, features)
        
        # Final predictions
        logits = self.classifier(attn_features)
        return_pred = self.return_head(logits)
        direction_pred = self.direction_head(logits)
        
        return return_pred.squeeze(-1), direction_pred.squeeze(-1)

class StockPredictionTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Multi-task loss: MSE for returns + BCE for direction
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            price = batch['price'].to(self.device)
            fundamental = batch['fundamental'].to(self.device)
            sentiment = batch['sentiment'].to(self.device)
            actions = batch['actions'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            return_pred, direction_pred = self.model(price, fundamental, sentiment, actions)
            
            # Calculate losses
            return_loss = self.mse_loss(return_pred, targets[:, 0])
            direction_loss = self.bce_loss(direction_pred, targets[:, 1])
            
            # Combined loss
            total_batch_loss = return_loss + direction_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        return_predictions = []
        direction_predictions = []
        true_returns = []
        true_directions = []
        
        with torch.no_grad():
            for batch in dataloader:
                price = batch['price'].to(self.device)
                fundamental = batch['fundamental'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                actions = batch['actions'].to(self.device)
                targets = batch['target'].to(self.device)
                
                return_pred, direction_pred = self.model(price, fundamental, sentiment, actions)
                
                # Calculate losses
                return_loss = self.mse_loss(return_pred, targets[:, 0])
                direction_loss = self.bce_loss(direction_pred, targets[:, 1])
                total_loss += (return_loss + direction_loss).item()
                
                # Store predictions
                return_predictions.extend(return_pred.cpu().numpy())
                direction_predictions.extend(direction_pred.cpu().numpy())
                true_returns.extend(targets[:, 0].cpu().numpy())
                true_directions.extend(targets[:, 1].cpu().numpy())
        
        # Calculate metrics
        return_mse = np.mean((np.array(return_predictions) - np.array(true_returns)) ** 2)
        direction_accuracy = np.mean((np.array(direction_predictions) > 0.5) == np.array(true_directions))
        
        return total_loss / len(dataloader), return_mse, direction_accuracy
    
    def train(self, train_loader, val_loader, num_epochs=100):
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_mse, val_acc = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, MSE: {val_mse:.6f}, Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

def main():
    """Main training pipeline"""
    
    # Initialize data processor
    processor = AAPLDataProcessor()
    
    # File paths - update these with your actual file paths
    file_paths = {
        'balance_sheet': 'AAPL_balance_sheet.xlsx',
        'income_statement': 'AAPL_income_statement.xlsx',
        'cash_flow': 'AAPL_cash_flow.xlsx',
        'news_data': 'AAPL_news_20250803.xlsx',
        # Add other file paths as needed
    }
    
    # Load and process data
    print("Loading and processing data...")
    data = processor.load_and_process_data(file_paths)
    print(f"Processed data shape: {data.shape}")
    
    # Create dataset
    dataset = AAPLDataset(data, sequence_length=60, prediction_horizon=5)
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = MultiModalStockPredictor(
        price_features=len(dataset.price_features),
        fundamental_features=len(dataset.fundamental_features),
        sentiment_features=len(dataset.sentiment_features),
        action_features=len(dataset.action_features),
        hidden_dim=128
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = StockPredictionTrainer(model)
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    print("Training completed!")

if __name__ == "__main__":
    main()