import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from nothingbutnet.data_collector import NBADataCollector
from nothingbutnet.models.spread_predictor import SpreadPredictor
from nothingbutnet.database import init_db, get_session, Game, Team, TeamStats

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs/training', exist_ok=True)
    
    # Set up logging with timestamp in filename
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
    """Fetch game data from the database"""
    logging.info("Fetching data from database...")
    
    # Initialize database connection
    try:
        init_db()
        session = get_session()
        logging.info("Successfully connected to database")
    except Exception as e:
        logging.error(f"Failed to connect to database: {str(e)}")
        raise
    
    try:
        # Fetch all games
        logging.info("Fetching games...")
        games_query = session.query(Game).all()
        num_games = len(games_query)
        logging.info(f"Found {num_games} games in database")
        
        if num_games == 0:
            logging.error("No games found in database. Please run data collection first.")
            raise ValueError("Database is empty - no games found")
        
        # Convert to DataFrame
        games_data = []
        for game in games_query:
            try:
                home_team = session.query(Team).filter(Team.id == game.home_team_id).first()
                away_team = session.query(Team).filter(Team.id == game.away_team_id).first()
                
                if not home_team or not away_team:
                    logging.warning(f"Missing team data for game {game.id}: home_team={bool(home_team)}, away_team={bool(away_team)}")
                    continue
                
                # Only include completed games with scores
                if game.home_score is not None and game.away_score is not None:
                    game_dict = {
                        'date': game.date,
                        'home_team_id': game.home_team_id,
                        'away_team_id': game.away_team_id,
                        'home_team_name': home_team.name,
                        'away_team_name': away_team.name,
                        'home_team_points': game.home_score,
                        'away_team_points': game.away_score,
                        'home_team_won': game.home_score > game.away_score,
                        'point_differential': game.home_score - game.away_score,
                        'season': game.season,
                        'game_id': game.id,
                        'team_id': game.home_team_id,  # For home team perspective
                        'points': game.home_score,
                        'points_allowed': game.away_score,
                        'won': 1 if game.home_score > game.away_score else 0,
                        # Add placeholder values for required columns
                        'fg_percentage': 0.45,  # Placeholder
                        'three_pt_percentage': 0.35,  # Placeholder
                        'ft_percentage': 0.75,  # Placeholder
                        'offensive_rebounds': 10,  # Placeholder
                        'defensive_rebounds': 30,  # Placeholder
                        'assists': 20,  # Placeholder
                        'turnovers': 15,  # Placeholder
                        'steals': 8,  # Placeholder
                        'blocks': 5,  # Placeholder
                        'fg_attempts': 85,  # Placeholder
                        'fg_made': 38,  # Placeholder
                        'three_pt_made': 12,  # Placeholder
                        'ft_attempts': 20,  # Placeholder
                        'opp_offensive_rebounds': 9,  # Placeholder
                        'opp_defensive_rebounds': 32,  # Placeholder
                    }
                    games_data.append(game_dict)
                    
                    # Add away team perspective as well
                    away_game_dict = game_dict.copy()
                    away_game_dict.update({
                        'team_id': game.away_team_id,
                        'points': game.away_score,
                        'points_allowed': game.home_score,
                        'won': 1 if game.away_score > game.home_score else 0,
                        'point_differential': game.away_score - game.home_score
                    })
                    games_data.append(away_game_dict)
            except Exception as e:
                logging.error(f"Error processing game {game.id}: {str(e)}")
                continue
        
        if not games_data:
            logging.error("No valid games data could be processed")
            raise ValueError("No valid games data available")
            
        games_df = pd.DataFrame(games_data)
        logging.info(f"Successfully processed {len(games_df)} game records")
        
        # Fetch team stats
        logging.info("Fetching team stats...")
        team_stats_query = session.query(TeamStats).all()
        num_stats = len(team_stats_query)
        logging.info(f"Found {num_stats} team stat records in database")
        
        if num_stats == 0:
            logging.warning("No team stats found in database. Will proceed with limited features.")
        
        team_stats_data = []
        for stat in team_stats_query:
            try:
                stat_dict = {
                    'team_id': stat.team_id,
                    'date': stat.date,
                    'season': stat.season,
                    'offensive_rating': stat.offensive_rating if stat.offensive_rating else 100.0,
                    'defensive_rating': stat.defensive_rating if stat.defensive_rating else 100.0,
                    'net_rating': stat.net_rating if stat.net_rating else 0.0,
                    'pace': stat.pace if stat.pace else 100.0,
                    'games_played': stat.games_played if stat.games_played else 0,
                    'wins': stat.wins if stat.wins else 0,
                    'losses': stat.losses if stat.losses else 0,
                    'position': 0,  # Will be calculated later
                    'games_behind': 0  # Will be calculated later
                }
                team_stats_data.append(stat_dict)
            except Exception as e:
                logging.error(f"Error processing team stats for team {stat.team_id}: {str(e)}")
                continue
        
        team_stats_df = pd.DataFrame(team_stats_data)
        logging.info(f"Successfully processed {len(team_stats_df)} team stat records")
        
        # Create a simple standings DataFrame based on team stats
        try:
            standings_df = team_stats_df.copy()
            if 'season' not in standings_df.columns:
                logging.warning("No season column found in team stats. Adding default season.")
                standings_df['season'] = datetime.now().year
            
            standings_df['position'] = standings_df.groupby(['season'])['wins'].rank(ascending=False)
            standings_df['games_behind'] = standings_df.groupby(['season'])['wins'].transform('max') - standings_df['wins']
            logging.info("Successfully calculated standings")
        except Exception as e:
            logging.error(f"Error calculating standings: {str(e)}")
            standings_df = pd.DataFrame()  # Empty DataFrame as fallback
        
        # Create synthetic player stats
        logging.info("Creating synthetic player stats...")
        teams = games_df['team_id'].unique()
        dates = sorted(games_df['date'].unique())
        
        player_data = []
        player_id = 1
        
        try:
            for team_id in teams:
                # Create 10 players per team
                for i in range(10):
                    is_starter = i < 5  # First 5 are starters
                    
                    # Create stats for each date
                    for date in dates:
                        # Only include dates where the team played
                        if date in games_df[games_df['team_id'] == team_id]['date'].values:
                            minutes = np.random.normal(30 if is_starter else 15, 5)
                            minutes = max(0, min(48, minutes))
                            
                            points = np.random.normal(15 if is_starter else 6, 5)
                            points = max(0, points)
                            
                            player_dict = {
                                'team_id': team_id,
                                'player_id': player_id,
                                'name': f"Player_{player_id}",
                                'date': date,
                                'minutes': minutes,
                                'points': points,
                                'rebounds': np.random.normal(6 if is_starter else 2, 2),
                                'assists': np.random.normal(4 if is_starter else 1, 2),
                                'is_available': True,
                                'plus_minus': np.random.normal(5 if is_starter else 0, 10),
                                'player_efficiency': np.random.normal(15 if is_starter else 10, 5)
                            }
                            player_data.append(player_dict)
                    
                    player_id += 1
            
            players_df = pd.DataFrame(player_data)
            logging.info(f"Successfully created synthetic data for {len(teams)} teams and {player_id-1} players")
            
        except Exception as e:
            logging.error(f"Error creating synthetic player data: {str(e)}")
            players_df = pd.DataFrame()  # Empty DataFrame as fallback
        
        return games_df, team_stats_df, players_df, standings_df
        
    except Exception as e:
        logging.error(f"Unexpected error in fetch_data_from_db: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()
        logging.info("Database session closed")

def main():
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting model training pipeline")
    logging.info(f"Logs will be saved to: {log_file}")
    
    # Create necessary directories
    try:
        for directory in ['data/raw', 'data/processed', 'data/models']:
            os.makedirs(directory, exist_ok=True)
        logging.info("Successfully created required directories")
    except Exception as e:
        logging.error(f"Failed to create directories: {str(e)}")
        raise
    
    try:
        # Fetch data from database
        logging.info("Starting data fetch from database...")
        games_df, team_stats_df, players_df, standings_df = fetch_data_from_db()
        
        # Log data shapes and basic info
        logging.info(f"Data fetch complete:")
        logging.info(f"- Games data shape: {games_df.shape}")
        logging.info(f"- Team stats shape: {team_stats_df.shape}")
        logging.info(f"- Players data shape: {players_df.shape}")
        logging.info(f"- Standings data shape: {standings_df.shape}")
        
        # Save raw data for reference
        logging.info("Saving raw data...")
        try:
            games_df.to_parquet('data/raw/games_from_db.parquet')
            team_stats_df.to_parquet('data/raw/team_stats_from_db.parquet')
            standings_df.to_parquet('data/raw/standings_from_db.parquet')
            players_df.to_parquet('data/raw/players_synthetic.parquet')
            logging.info("Successfully saved raw data files")
        except Exception as e:
            logging.error(f"Error saving raw data: {str(e)}")
            raise
        
        # Check if we have enough data
        min_games_required = 1000
        if len(games_df) < min_games_required:
            logging.warning(f"Only {len(games_df)} games available. Model may not perform well with limited data.")
            logging.warning(f"Recommended minimum: {min_games_required} games")
        else:
            logging.info(f"Training with {len(games_df)} games")
        
        # Initialize predictor and train model
        logging.info("Initializing SpreadPredictor...")
        try:
            predictor = SpreadPredictor()
            
            # Train model
            logging.info("Starting model training...")
            history = predictor.train(games_df, players_df, standings_df, 
                                   batch_size=32, epochs=100, patience=10)
            
            # Save model
            model_path = 'data/models/spread_predictor.pt'
            logging.info(f"Saving model to {model_path}")
            predictor.save_model(model_path)
            
            # Log training results
            logging.info("Training complete!")
            logging.info(f"Final Results:")
            logging.info(f"- Test MAE: {history['test_mae']:.1f} points")
            logging.info(f"- Final training loss: {history['train_losses'][-1]:.4f}")
            logging.info(f"- Final validation loss: {history['val_losses'][-1]:.4f}")
            
            # Log model performance metrics
            if 'metrics' in history:
                for metric_name, value in history['metrics'].items():
                    logging.info(f"- {metric_name}: {value:.4f}")
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}", exc_info=True)
            raise
        
    except Exception as e:
        logging.error(f"Fatal error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Training failed", exc_info=True)
        raise 