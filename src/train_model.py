import os
import logging
from NBADataCollector import NBADataCollector
from NBASpreadPredictor import NBASpreadPredictor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training/training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Set up logging
    setup_logging()
    logging.info("Starting model training pipeline")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('logs/training', exist_ok=True)
    
    try:
        # Initialize collector and fetch data
        collector = NBADataCollector()
        
        logging.info("Fetching game data...")
        games_df, advanced_df = collector.fetch_basketball_reference()
        
        logging.info("Fetching player stats...")
        players_df = collector.fetch_player_stats(2024)
        
        logging.info("Fetching standings...")
        standings_df = collector.fetch_standings()
        
        # Save raw data
        games_df.to_parquet('data/raw/games.parquet')
        advanced_df.to_parquet('data/raw/advanced.parquet')
        players_df.to_parquet('data/raw/players.parquet')
        standings_df.to_parquet('data/raw/standings.parquet')
        
        # Initialize predictor and train model
        logging.info("Training model...")
        predictor = NBASpreadPredictor()
        
        # Validate data
        predictor.validate_data(games_df, players_df, standings_df)
        
        # Train model
        history, test_mae = predictor.train(games_df, players_df, standings_df)
        
        # Save model
        predictor.save_model()
        
        logging.info(f"Training complete! Test MAE: {test_mae:.1f} points")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 