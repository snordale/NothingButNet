#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from nothingbutnet.database import init_db, get_session, Game, TeamStats
from nothingbutnet.models.simple_spread_predictor import SpreadPredictor, NBADataset

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

def fetch_data_from_db():
    """Fetch and prepare data from the database"""
    logging.info("Fetching data from database...")
    
    # Initialize database connection
    session = get_session()
    
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
                'last_10_wins': stat.last_10_wins if stat.last_10_wins else 5,
                'offensive_rating': stat.offensive_rating if stat.offensive_rating else 100.0,
                'defensive_rating': stat.defensive_rating if stat.defensive_rating else 100.0,
                'net_rating': stat.net_rating if stat.net_rating else 0.0,
                'offensive_rebounds': stat.offensive_rebounds if stat.offensive_rebounds else 10.0,
                'defensive_rebounds': stat.defensive_rebounds if stat.defensive_rebounds else 30.0,
                'turnovers': stat.turnovers if stat.turnovers else 15.0
            }
            stats_data.append(stat_dict)
        
        stats_df = pd.DataFrame(stats_data)
        
        return games_df, stats_df
    
    finally:
        session.close()

def main():
    # Setup
    log_file = setup_logging()
    logging.info("Starting model training pipeline")
    logging.info(f"Logs will be saved to: {log_file}")
    
    try:
        # Create necessary directories
        os.makedirs('data/models', exist_ok=True)
        
        # Initialize database
        init_db()
        
        # Fetch and prepare data
        games_df, stats_df = fetch_data_from_db()
        
        if games_df.empty:
            raise ValueError("No games data available for training")
        
        # Initialize predictor
        predictor = SpreadPredictor()
        
        # Prepare features and targets
        features, targets = predictor.prepare_game_features(games_df, stats_df)
        
        if len(features) == 0:
            raise ValueError("No valid features could be created from the data")
        
        # Split data and scale features
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        train_features, val_features, train_targets, val_targets = train_test_split(
            train_features, train_targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        val_features_scaled = scaler.transform(val_features)
        test_features_scaled = scaler.transform(test_features)
        predictor.scaler = scaler
        
        # Create data loaders
        train_dataset = NBADataset(pd.DataFrame(train_features_scaled), train_targets)
        val_dataset = NBADataset(pd.DataFrame(val_features_scaled), val_targets)
        test_dataset = NBADataset(pd.DataFrame(test_features_scaled), test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Training on device: {device}")
        
        input_dim = train_features.shape[1]
        history = predictor.train(train_loader, val_loader, input_dim)
        
        # Evaluate on test set
        test_predictions = predictor.predict_spread(pd.DataFrame(test_features_scaled))
        predictions = test_predictions['spread']
        feature_importance = test_predictions['feature_importance']
        
        mae = np.mean(np.abs(predictions - test_targets))
        mse = np.mean((predictions - test_targets) ** 2)
        rmse = np.sqrt(mse)
        
        logging.info(f"Test Results:")
        logging.info(f"Best Validation Loss: {history['best_val_loss']:.4f}")
        logging.info(f"Test MAE: {mae:.4f}")
        logging.info(f"Test RMSE: {rmse:.4f}")
        logging.info("\nMost Important Features:")
        for feature, importance in feature_importance:
            logging.info(f"{feature}: {importance:.4f}")
        
        # Save model
        predictor.save_model('data/models/nba_predictor.pt')
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