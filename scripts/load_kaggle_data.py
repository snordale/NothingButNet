#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from nothingbutnet.database import init_db, save_game, save_team_stats, upsert_team
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_kaggle_auth():
    """Set up Kaggle authentication using environment variables"""
    load_dotenv()
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

def download_dataset():
    """Download the NBA dataset from Kaggle"""
    try:
        os.system('kaggle datasets download -d wyattowalsh/basketball -p data/kaggle')
        os.system('cd data/kaggle && unzip -o basketball.zip')
        logging.info("Dataset downloaded and extracted successfully")
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        raise

def calculate_rest_days(games_df, team_id, game_date):
    """Calculate the number of rest days for a team before a game."""
    # Convert game_date to datetime if it's not already
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)

    # Get previous games for this team
    previous_games = games_df[
        ((games_df['team_id_home'] == team_id) | (games_df['team_id_away'] == team_id)) &
        (games_df['game_date'] < game_date)
    ]

    if previous_games.empty:
        return None  # No previous games

    # Get the most recent game date
    last_game_date = pd.to_datetime(previous_games['game_date'].max())
    
    # Calculate days between games
    rest_days = (game_date - last_game_date).days - 1  # -1 because we don't count game days
    return max(0, rest_days)  # Ensure non-negative

def calculate_team_stats(games_df, team_id, game_date):
    """Calculate team statistics up to a given date."""
    # Get all previous games for this team
    prev_games = games_df[
        ((games_df['team_id_home'] == team_id) | (games_df['team_id_away'] == team_id)) &
        (games_df['game_date'] < game_date)
    ]
    
    if prev_games.empty:
        return {}  # No previous games
    
    # Calculate basic stats
    total_games = len(prev_games)
    
    wins = 0
    home_wins = 0
    away_wins = 0
    points_scored = 0
    points_allowed = 0
    
    # Last 10 games stats
    last_10_games = prev_games.tail(10)
    last_10_wins = 0
    last_10_points_scored = 0
    last_10_points_allowed = 0
    
    for _, game in prev_games.iterrows():
        is_home = game['team_id_home'] == team_id
        team_score = game['pts_home'] if is_home else game['pts_away']
        opp_score = game['pts_away'] if is_home else game['pts_home']
        
        # Update total stats
        if (is_home and team_score > opp_score) or (not is_home and team_score > opp_score):
            wins += 1
            if is_home:
                home_wins += 1
            else:
                away_wins += 1
        
        points_scored += team_score
        points_allowed += opp_score
    
    # Calculate last 10 games stats
    for _, game in last_10_games.iterrows():
        is_home = game['team_id_home'] == team_id
        team_score = game['pts_home'] if is_home else game['pts_away']
        opp_score = game['pts_away'] if is_home else game['pts_home']
        
        if (is_home and team_score > opp_score) or (not is_home and team_score > opp_score):
            last_10_wins += 1
        
        last_10_points_scored += team_score
        last_10_points_allowed += opp_score
    
    # Return stats dictionary
    return {
        'games_played': total_games,
        'wins': wins,
        'losses': total_games - wins,
        'home_wins': home_wins,
        'home_losses': len(prev_games[prev_games['team_id_home'] == team_id]) - home_wins,
        'away_wins': away_wins,
        'away_losses': len(prev_games[prev_games['team_id_away'] == team_id]) - away_wins,
        'points_per_game': points_scored / total_games,
        'points_allowed_per_game': points_allowed / total_games,
        'win_pct': wins / total_games,
        'last_10_wins': last_10_wins,
        'last_10_losses': 10 - last_10_wins if len(last_10_games) == 10 else len(last_10_games) - last_10_wins,
        'last_10_points_scored': last_10_points_scored,
        'last_10_points_allowed': last_10_points_allowed,
        'streak': 0  # We'll calculate this later if needed
    }

def determine_game_type(season_type):
    """Determine the game type based on the season type."""
    season_type = season_type.lower()
    if 'playoff' in season_type:
        return 'playoffs'
    elif 'regular' in season_type:
        return 'regular_season'
    else:
        return 'preseason'

def load_data():
    """Load data from Kaggle dataset into the database."""
    logging.info("Loading data from Kaggle dataset...")
    
    # Read games data
    games_df = pd.read_csv('data/kaggle/csv/game.csv')
    game_info_df = pd.read_csv('data/kaggle/csv/game_info.csv')
    line_score_df = pd.read_csv('data/kaggle/csv/line_score.csv')
    
    # Log unique season IDs and their date ranges to understand the data
    season_info = games_df.groupby('season_id').agg({
        'game_date': ['min', 'max']
    }).reset_index()
    logging.info("Season ID mappings:")
    for _, season in season_info.iterrows():
        logging.info(f"Season ID {season['season_id']}: {season['game_date']['min']} to {season['game_date']['max']}")
    
    # Convert date columns
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    
    # Map season IDs to actual years based on the game date
    games_df['season_year'] = games_df['game_date'].dt.year
    # If game is in Oct-Dec, it's part of the season that starts in that year
    # If game is in Jan-Jun, it's part of the season that started the previous year
    games_df['season_year'] = np.where(
        games_df['game_date'].dt.month >= 10,
        games_df['game_date'].dt.year,
        games_df['game_date'].dt.year - 1
    )
    
    # Filter games since 2013
    cutoff_date = pd.Timestamp('2013-01-01')
    games_df = games_df[games_df['game_date'] >= cutoff_date]
    logging.info(f"Found {len(games_df)} games since {cutoff_date.date()}")
    
    # Sort games by date
    games_df = games_df.sort_values('game_date')
    
    # Initialize database
    session = init_db()
    
    try:
        # Process each game
        for _, game in games_df.iterrows():
            try:
                # Get team names
                home_team = game['team_name_home']
                away_team = game['team_name_away']
                
                # Calculate rest days
                home_rest_days = calculate_rest_days(games_df, game['team_id_home'], game['game_date'])
                away_rest_days = calculate_rest_days(games_df, game['team_id_away'], game['game_date'])
                
                # Calculate team stats
                home_stats = calculate_team_stats(games_df, game['team_id_home'], game['game_date'])
                away_stats = calculate_team_stats(games_df, game['team_id_away'], game['game_date'])
                
                # Determine game type
                game_type = determine_game_type(game['season_type'])
                
                # Save game to database
                save_game(
                    session=session,
                    date=game['game_date'],
                    season=game['season_year'],  # Use our calculated season year instead of season_id
                    home_team=home_team,
                    away_team=away_team,
                    home_score=game['pts_home'],
                    away_score=game['pts_away'],
                    game_type=game_type,
                    rest_days_home=home_rest_days,
                    rest_days_away=away_rest_days
                )
                
                # Save team stats if available
                if home_stats:
                    save_team_stats(session, home_team, game['game_date'], game['season_year'], home_stats)
                if away_stats:
                    save_team_stats(session, away_team, game['game_date'], game['season_year'], away_stats)
                
                logging.info(f"Processed game: {away_team} @ {home_team} ({game['game_date'].date()})")
                
            except Exception as e:
                logging.error(f"Error processing game: {str(e)}")
                continue
        
        # Commit all changes
        session.commit()
        logging.info("Successfully loaded all data into database")
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        session.rollback()
        raise
    
    finally:
        session.close()

def main():
    setup_logging()
    setup_kaggle_auth()
    
    # Load data into database
    load_data()

if __name__ == "__main__":
    main() 