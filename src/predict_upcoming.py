import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from sqlalchemy import desc

from nothingbutnet.database import init_db, get_session, Game, Team, TeamStats
from nothingbutnet.models.spread_predictor import SpreadPredictor, SpreadPredictorNet
from nothingbutnet.data_collector import NBADataCollector

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs/predictions', exist_ok=True)
    
    # Set up logging with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/predictions/predictions_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def fetch_upcoming_games(days=7):
    """Fetch upcoming games from ESPN"""
    logging.info(f"Fetching upcoming games for the next {days} days...")
    
    # Initialize database connection
    init_db()
    session = get_session()
    
    try:
        # Use the data collector to fetch upcoming games
        collector = NBADataCollector()
        
        # Calculate the date range
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days)
        
        # Fetch upcoming games from ESPN
        upcoming_games = []
        
        current_date = start_date
        while current_date <= end_date:
            try:
                # Format date for ESPN API
                date_str = current_date.strftime("%Y%m%d")
                
                # Fetch games for this date
                games = collector.fetch_espn_games(date_str)
                
                if games:
                    for game in games:
                        # Check if teams exist in our database
                        home_team = session.query(Team).filter(Team.espn_id == game['home_team_id']).first()
                        away_team = session.query(Team).filter(Team.espn_id == game['away_team_id']).first()
                        
                        if home_team and away_team:
                            game_info = {
                                'date': current_date,
                                'home_team_id': home_team.id,
                                'away_team_id': away_team.id,
                                'home_team_name': home_team.name,
                                'away_team_name': away_team.name,
                                'espn_id': game.get('espn_id'),
                                'season': datetime.now().year if datetime.now().month > 6 else datetime.now().year - 1
                            }
                            upcoming_games.append(game_info)
                
                logging.info(f"Found {len(games) if games else 0} games for {current_date}")
                
            except Exception as e:
                logging.error(f"Error fetching games for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        logging.info(f"Found {len(upcoming_games)} upcoming games")
        
        # Convert to DataFrame
        upcoming_df = pd.DataFrame(upcoming_games) if upcoming_games else pd.DataFrame()
        
        # If no upcoming games found, create a sample for testing
        if upcoming_df.empty:
            logging.warning("No upcoming games found. Creating sample games for testing.")
            
            # Get all teams
            teams = session.query(Team).all()
            team_ids = [team.id for team in teams]
            team_names = [team.name for team in teams]
            
            # Create sample games
            sample_games = []
            for i in range(0, len(team_ids), 2):
                if i + 1 < len(team_ids):
                    game_date = start_date + timedelta(days=i % days)
                    sample_games.append({
                        'date': game_date,
                        'home_team_id': team_ids[i],
                        'away_team_id': team_ids[i+1],
                        'home_team_name': team_names[i],
                        'away_team_name': team_names[i+1],
                        'espn_id': f"sample_{i}",
                        'season': datetime.now().year if datetime.now().month > 6 else datetime.now().year - 1
                    })
            
            upcoming_df = pd.DataFrame(sample_games)
            logging.info(f"Created {len(sample_games)} sample games for testing")
        
        # Fetch historical games for feature preparation
        historical_games_query = session.query(Game).filter(
            Game.date < start_date,
            Game.home_score.isnot(None),
            Game.away_score.isnot(None)
        ).order_by(desc(Game.date)).limit(1000).all()
        
        logging.info(f"Fetched {len(historical_games_query)} historical games for feature preparation")
        
        # Convert to DataFrame
        historical_games_data = []
        for game in historical_games_query:
            game_dict = {
                'date': game.date,
                'home_team_id': game.home_team_id,
                'away_team_id': game.away_team_id,
                'home_team_points': game.home_score,
                'away_team_points': game.away_score,
                'home_team_won': game.home_score > game.away_score if game.home_score and game.away_score else None,
                'point_differential': game.home_score - game.away_score if game.home_score and game.away_score else None,
                'season': game.season,
                'game_id': game.id,
                # Add additional columns needed by the model
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
            historical_games_data.append(game_dict)
            
            # Add away team perspective as well
            away_game_dict = game_dict.copy()
            away_game_dict.update({
                'team_id': game.away_team_id,
                'points': game.away_score,
                'points_allowed': game.home_score,
                'won': 1 if game.away_score > game.home_score else 0,
                'point_differential': game.away_score - game.home_score
            })
            historical_games_data.append(away_game_dict)
        
        # Combine historical games with upcoming games (with empty scores)
        for _, game in upcoming_df.iterrows():
            # Add home team perspective
            home_dict = {
                'date': game['date'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'home_team_points': None,
                'away_team_points': None,
                'home_team_won': None,
                'point_differential': None,
                'season': game['season'],
                'game_id': game.get('espn_id', f"upcoming_{len(historical_games_data)}"),
                'team_id': game['home_team_id'],
                'points': None,
                'points_allowed': None,
                'won': None,
                # Add placeholder values for required columns
                'fg_percentage': 0.45,
                'three_pt_percentage': 0.35,
                'ft_percentage': 0.75,
                'offensive_rebounds': 10,
                'defensive_rebounds': 30,
                'assists': 20,
                'turnovers': 15,
                'steals': 8,
                'blocks': 5,
                'fg_attempts': 85,
                'fg_made': 38,
                'three_pt_made': 12,
                'ft_attempts': 20,
                'opp_offensive_rebounds': 9,
                'opp_defensive_rebounds': 32,
            }
            historical_games_data.append(home_dict)
            
            # Add away team perspective
            away_dict = home_dict.copy()
            away_dict.update({
                'team_id': game['away_team_id'],
            })
            historical_games_data.append(away_dict)
        
        games_df = pd.DataFrame(historical_games_data)
        
        # Fetch team stats
        team_stats_query = session.query(TeamStats).all()
        
        team_stats_data = []
        for stat in team_stats_query:
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
        
        team_stats_df = pd.DataFrame(team_stats_data)
        
        # Create a simple standings DataFrame based on team stats
        standings_df = team_stats_df.copy()
        standings_df['position'] = standings_df.groupby(['season'])['wins'].rank(ascending=False)
        standings_df['games_behind'] = standings_df.groupby(['season'])['wins'].transform('max') - standings_df['wins']
        
        # Create synthetic player stats
        teams = games_df['team_id'].unique()
        
        # Convert all dates to datetime objects to ensure consistency
        games_df['date'] = games_df['date'].apply(lambda x: x if isinstance(x, datetime) else datetime.combine(x, datetime.min.time()))
        
        # Now sort the unique dates
        dates = sorted(games_df['date'].unique())
        
        player_data = []
        player_id = 1
        
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
        
        return upcoming_df, games_df, team_stats_df, players_df, standings_df
        
    finally:
        session.close()

def predict_upcoming_games():
    """Predict upcoming NBA games"""
    log_file = setup_logging()
    logging.info("Starting prediction pipeline for upcoming games")
    logging.info(f"Logs will be saved to: {log_file}")
    
    try:
        # Fetch upcoming games and historical data
        upcoming_df, games_df, team_stats_df, players_df, standings_df = fetch_upcoming_games(days=7)
        
        if upcoming_df.empty:
            logging.error("No upcoming games found to predict")
            return
        
        # Ensure all dates are datetime objects
        logging.info("Converting all dates to datetime objects for consistency...")
        
        # Convert dates in upcoming_df
        if 'date' in upcoming_df.columns:
            upcoming_df['date'] = pd.to_datetime(upcoming_df['date'])
        
        # Convert dates in games_df
        if 'date' in games_df.columns:
            games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Convert dates in team_stats_df
        if 'date' in team_stats_df.columns:
            team_stats_df['date'] = pd.to_datetime(team_stats_df['date'])
        
        # Convert dates in players_df
        if 'date' in players_df.columns:
            players_df['date'] = pd.to_datetime(players_df['date'])
        
        # Convert dates in standings_df
        if 'date' in standings_df.columns:
            standings_df['date'] = pd.to_datetime(standings_df['date'])
        
        # Load the trained model
        model_path = 'data/models/spread_predictor.pt'
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return
        
        logging.info(f"Loading model from {model_path}")
        
        # Create a custom model loading function to handle PyTorch 2.6+ weights_only issue
        predictor = SpreadPredictor()
        
        # Add StandardScaler to safe globals for torch.load
        try:
            from sklearn.preprocessing import StandardScaler
            import torch.serialization
            
            # Now load the model with weights_only=False since this is a trusted file
            checkpoint = torch.load(model_path, weights_only=False)
            input_dim = len(checkpoint['scaler_state'].mean_)
            
            # Set up the model manually
            predictor.model = SpreadPredictorNet(input_dim).to(predictor.device)
            predictor.model.load_state_dict(checkpoint['model_state_dict'])
            predictor.scaler = checkpoint['scaler_state']
            
            logging.info("Successfully loaded model with custom loading approach")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            return
        
        # Prepare features for prediction
        logging.info("Preparing game features for prediction...")
        
        # Create a dictionary to store predictions
        predictions = []
        
        # Ensure all dates in games_df are datetime objects
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Process each upcoming game
        for _, game in upcoming_df.iterrows():
            try:
                home_team_id = game['home_team_id']
                away_team_id = game['away_team_id']
                game_date = game['date']
                
                # Ensure game_date is a datetime object
                if isinstance(game_date, datetime):
                    game_datetime = game_date
                else:
                    game_datetime = datetime.combine(game_date, datetime.min.time())
                
                # We need to use the full games_df for historical context, but we're only predicting this specific game
                # First, filter the games_df to include only games before the current game date
                historical_games = games_df[games_df['date'] < game_datetime].copy()
                
                # Create a mini-dataset with just this game
                game_df = pd.DataFrame([{
                    'date': game_datetime,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_team_points': None,
                    'away_team_points': None,
                    'game_id': f"upcoming_{home_team_id}_{away_team_id}"
                }])
                
                # Add the current game to the historical games
                prediction_games = pd.concat([historical_games, game_df])
                
                # Prepare features
                X, _ = predictor.prepare_game_features(prediction_games, players_df, standings_df)
                
                # Make prediction - use the last row which corresponds to our current game
                predicted_spread = predictor.predict_spread(X.iloc[[-1]])
                
                # Calculate win probability based on spread
                # Using a simple logistic function to convert spread to probability
                win_probability = 1 / (1 + np.exp(-predicted_spread * 0.1))
                
                # Store prediction
                predictions.append({
                    'date': game_date,
                    'home_team': game['home_team_name'],
                    'away_team': game['away_team_name'],
                    'predicted_spread': predicted_spread,
                    'home_win_probability': win_probability,
                    'away_win_probability': 1 - win_probability,
                    'predicted_winner': game['home_team_name'] if predicted_spread > 0 else game['away_team_name'],
                    'confidence': abs(win_probability - 0.5) * 2  # Scale from 0-1
                })
                
            except Exception as e:
                logging.error(f"Error predicting game {game['home_team_name']} vs {game['away_team_name']}: {e}")
                continue
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Save predictions
        os.makedirs('data/predictions', exist_ok=True)
        predictions_file = f'data/predictions/upcoming_games_{datetime.now().strftime("%Y%m%d")}.csv'
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"Predictions saved to {predictions_file}")
        
        # Print predictions
        print("\nUpcoming Game Predictions:")
        print(f"{'Date':<12} {'Home Team':<20} {'Away Team':<20} {'Predicted Spread':<15} {'Win Prob':<10} {'Confidence':<10}")
        print("-" * 90)
        
        for _, row in predictions_df.iterrows():
            home_win_prob = row['home_win_probability'] * 100
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['home_team']:<20} {row['away_team']:<20} "
                  f"{row['predicted_spread']:<15.1f} {home_win_prob:<10.1f}% {row['confidence']:<10.2f}")
        
        print(f"\nPredictions saved to {predictions_file}")
        
    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    predict_upcoming_games() 