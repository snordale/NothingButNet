import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from sqlalchemy import desc

from nothingbutnet.database import init_db, get_session, Game, Team, TeamStats
from nothingbutnet.models.spread_predictor import SpreadPredictor, SpreadPredictorNet

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs/testing', exist_ok=True)
    
    # Set up logging with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/testing/model_test_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def fetch_recent_games(days=30):
    """Fetch recent games from the database"""
    logging.info(f"Fetching games from the last {days} days...")
    
    # Initialize database connection
    init_db()
    session = get_session()
    
    try:
        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch games within the date range
        games_query = session.query(Game).filter(
            Game.date >= start_date,
            Game.date <= end_date,
            Game.home_score.isnot(None),  # Only completed games
            Game.away_score.isnot(None)
        ).order_by(desc(Game.date)).all()
        
        logging.info(f"Found {len(games_query)} recent games with scores")
        
        # If no recent games, fetch the most recent games with scores
        if not games_query:
            logging.info("No recent games found. Fetching the most recent games with scores...")
            games_query = session.query(Game).filter(
                Game.home_score.isnot(None),
                Game.away_score.isnot(None)
            ).order_by(desc(Game.date)).limit(20).all()
            
            logging.info(f"Found {len(games_query)} games with scores")
        
        # Convert to DataFrame
        games_data = []
        for game in games_query:
            home_team = session.query(Team).filter(Team.id == game.home_team_id).first()
            away_team = session.query(Team).filter(Team.id == game.away_team_id).first()
            
            game_dict = {
                'date': game.date,
                'home_team_id': game.home_team_id,
                'away_team_id': game.away_team_id,
                'home_team_name': home_team.name if home_team else None,
                'away_team_name': away_team.name if away_team else None,
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
        
        # Check if we have any games
        if not games_data:
            logging.error("No games found with scores. Cannot proceed with testing.")
            return None, None, None, None
            
        games_df = pd.DataFrame(games_data)
        
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
        
        return games_df, team_stats_df, players_df, standings_df
        
    finally:
        session.close()

def test_model_on_recent_games():
    """Test the trained model on recent games"""
    log_file = setup_logging()
    logging.info("Starting model testing on recent games")
    logging.info(f"Logs will be saved to: {log_file}")
    
    try:
        # Fetch recent games
        games_df, team_stats_df, players_df, standings_df = fetch_recent_games(days=30)
        
        # Check if we have data to work with
        if games_df is None or games_df.empty:
            logging.error("No game data available for testing. Exiting.")
            print("No game data available for testing. Please ensure there are games with scores in the database.")
            return
        
        # Load the trained model
        model_path = 'data/models/spread_predictor.pt'
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return
        
        logging.info(f"Loading model from {model_path}")
        
        # Get unique games (remove duplicate team perspectives)
        unique_games = games_df.drop_duplicates(subset=['game_id']).copy()
        
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
        
        # Create a dictionary to store results
        results = []
        
        # Ensure all dates in games_df are datetime objects
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Process each game
        for _, game in unique_games.iterrows():
            try:
                home_team_id = game['home_team_id']
                away_team_id = game['away_team_id']
                game_date = game['date']
                actual_spread = game['home_team_points'] - game['away_team_points']
                
                # Create a mini-dataset with just this game
                game_df = pd.DataFrame([game])
                
                # We need to use the full games_df for historical context, but we're only predicting this specific game
                # First, filter the games_df to include only games before the current game date
                historical_games = games_df[games_df['date'] < game_date].copy()
                
                # Add the current game to the historical games
                prediction_games = pd.concat([historical_games, game_df])
                
                # Prepare features
                X, _ = predictor.prepare_game_features(prediction_games, players_df, standings_df)
                
                # Make prediction - use the last row which corresponds to our current game
                predicted_spread = predictor.predict_spread(X.iloc[[-1]])
                
                # Store results
                results.append({
                    'date': game_date,
                    'home_team': game['home_team_name'],
                    'away_team': game['away_team_name'],
                    'actual_spread': actual_spread,
                    'predicted_spread': predicted_spread,
                    'error': abs(predicted_spread - actual_spread),
                    'correct_winner': (predicted_spread > 0 and actual_spread > 0) or 
                                     (predicted_spread < 0 and actual_spread < 0)
                })
                
            except Exception as e:
                logging.error(f"Error processing game {game['home_team_name']} vs {game['away_team_name']}: {e}")
                continue
        
        # Check if we have any results
        if not results:
            logging.error("No predictions could be made. Check the logs for errors.")
            print("No predictions could be made. Check the logs for errors.")
            return
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate performance metrics
        mae = results_df['error'].mean()
        winner_accuracy = results_df['correct_winner'].mean() * 100
        
        logging.info(f"Model performance on recent games:")
        logging.info(f"Mean Absolute Error: {mae:.2f} points")
        logging.info(f"Winner Prediction Accuracy: {winner_accuracy:.2f}%")
        
        # Save results
        os.makedirs('data/predictions', exist_ok=True)
        results_file = f'data/predictions/recent_games_test_{datetime.now().strftime("%Y%m%d")}.csv'
        results_df.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")
        
        # Print detailed results
        print("\nDetailed Results:")
        print(f"{'Date':<12} {'Home Team':<20} {'Away Team':<20} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Correct?':<8}")
        print("-" * 90)
        
        for _, row in results_df.iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['home_team']:<20} {row['away_team']:<20} "
                  f"{row['actual_spread']:<8.1f} {row['predicted_spread']:<10.1f} {row['error']:<8.1f} "
                  f"{str(row['correct_winner']):<8}")
        
        print("\nSummary:")
        print(f"Mean Absolute Error: {mae:.2f} points")
        print(f"Winner Prediction Accuracy: {winner_accuracy:.2f}%")
        
    except Exception as e:
        logging.error(f"Error in testing pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    test_model_on_recent_games() 