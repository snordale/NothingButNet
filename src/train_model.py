import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nothingbutnet.database import Game, Team, TeamStats, Base

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs/training', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

class NBADataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SimpleNBAPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNBAPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def fetch_data_from_db():
    """Fetch and prepare data from the database"""
    logging.info("Fetching data from database...")
    
    # Initialize database connection
    engine = create_engine('sqlite:///data/kaggle/nba.sqlite')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    
    try:
        # Fetch games and team stats
        games = session.query(Game).all()
        team_stats = session.query(TeamStats).all()
        
        # Convert to DataFrames
        games_data = []
        for game in games:
            if game.home_score is not None and game.away_score is not None:
                game_dict = {
                    'date': game.date,
                    'home_team_id': game.home_team_id,
                    'away_team_id': game.away_team_id,
                    'home_score': game.home_score,
                    'away_score': game.away_score,
                    'point_diff': game.home_score - game.away_score,
                    'season': game.season,
                    'rest_days_home': game.rest_days_home if game.rest_days_home else 1,
                    'rest_days_away': game.rest_days_away if game.rest_days_away else 1
                }
                games_data.append(game_dict)
        
        games_df = pd.DataFrame(games_data)
        
        # Process team stats
        stats_data = []
        for stat in team_stats:
            stat_dict = {
                'team_id': stat.team_id,
                'date': stat.date,
                'season': stat.season,
                'games_played': stat.games_played,
                'wins': stat.wins,
                'losses': stat.losses,
                'win_pct': stat.wins / (stat.games_played) if stat.games_played > 0 else 0.0,
                'points_per_game': stat.points_per_game if stat.points_per_game else 100.0,
                'points_allowed_per_game': stat.points_allowed_per_game if stat.points_allowed_per_game else 100.0,
                'last_10_wins': stat.last_10_wins if stat.last_10_wins else 5
            }
            stats_data.append(stat_dict)
        
        stats_df = pd.DataFrame(stats_data)
        
        return games_df, stats_df
    
    finally:
        session.close()

def prepare_features(games_df, stats_df):
    """Prepare features for model training"""
    features = []
    targets = []
    
    for _, game in games_df.iterrows():
        # Get team stats before the game
        home_stats = stats_df[
            (stats_df['team_id'] == game['home_team_id']) &
            (stats_df['date'] < game['date'])
        ].sort_values('date').iloc[-1] if not stats_df.empty else None
        
        away_stats = stats_df[
            (stats_df['team_id'] == game['away_team_id']) &
            (stats_df['date'] < game['date'])
        ].sort_values('date').iloc[-1] if not stats_df.empty else None
        
        if home_stats is not None and away_stats is not None:
            feature_vector = [
                home_stats['win_pct'],
                home_stats['points_per_game'],
                home_stats['points_allowed_per_game'],
                home_stats['last_10_wins'],
                away_stats['win_pct'],
                away_stats['points_per_game'],
                away_stats['points_allowed_per_game'],
                away_stats['last_10_wins'],
                game['rest_days_home'],
                game['rest_days_away'],
                1.0  # Home court advantage indicator
            ]
            
            features.append(feature_vector)
            targets.append(game['point_diff'])
    
    return np.array(features), np.array(targets)

def train_model(train_loader, val_loader, input_dim, device, epochs=100, patience=10):
    """Train the model with early stopping"""
    model = SimpleNBAPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs.squeeze(), targets).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logging.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def main():
    # Setup
    log_file = setup_logging()
    logging.info(f"Logs will be saved to: {log_file}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        # Create directories
        os.makedirs('data/models', exist_ok=True)
        
        # Fetch and prepare data
        games_df, stats_df = fetch_data_from_db()
        features, targets = prepare_features(games_df, stats_df)
        
        # Split data
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        train_features, val_features, train_targets, val_targets = train_test_split(
            train_features, train_targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)
        
        # Create data loaders
        train_dataset = NBADataset(train_features, train_targets)
        val_dataset = NBADataset(val_features, val_targets)
        test_dataset = NBADataset(test_features, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Train model
        input_dim = train_features.shape[1]
        model = train_model(train_loader, val_loader, input_dim, device)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        criterion = nn.MSELoss()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                test_loss += criterion(outputs.squeeze(), targets).item()
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        test_loss /= len(test_loader)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        logging.info(f"Test Results:")
        logging.info(f"MSE: {test_loss:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_dim': input_dim,
            'test_mae': mae
        }, 'data/models/nba_predictor.pt')
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Training failed", exc_info=True)
        raise 