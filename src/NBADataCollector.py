import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import os
import kaggle
import logging

class NBADataCollector:
    def __init__(self):
        self.base_url = "https://www.basketball-reference.com"
        self.seasons = range(2015, 2024)  # Last ~10 seasons
        
    def fetch_basketball_reference(self):
        """Fetch data from Basketball Reference"""
        all_games = []
        all_advanced = []
        
        for season in self.seasons:
            print(f"Fetching season {season}-{season+1}")
            
            # Get schedule/results
            schedule_url = f"{self.base_url}/leagues/NBA_{season}_games.html"
            schedule_df = pd.read_html(schedule_url)[0]
            
            # Get team advanced stats
            advanced_url = f"{self.base_url}/leagues/NBA_{season}_ratings.html"
            advanced_df = pd.read_html(advanced_url)[0]
            advanced_df['Season'] = season
            
            all_games.append(schedule_df)
            all_advanced.append(advanced_df)
            
            # Be nice to the server
            time.sleep(3)
        
        games_df = pd.concat(all_games)
        advanced_df = pd.concat(all_advanced)
        
        return self.process_basketball_reference_data(games_df, advanced_df)
    
    def process_basketball_reference_data(self, games_df, advanced_df):
        """Clean and process Basketball Reference data"""
        # Clean games data
        games_df = games_df.rename(columns={
            'Visitor/Neutral': 'Away',
            'Home/Neutral': 'Home',
            'PTS.1': 'Home_Points',
            'PTS': 'Away_Points',
        })
        
        # Calculate point spread (from home team perspective)
        games_df['Point_Spread'] = games_df['Home_Points'] - games_df['Away_Points']
        
        # Convert date
        games_df['Date'] = pd.to_datetime(games_df['Date'])
        
        # Clean advanced stats
        advanced_df = advanced_df.dropna(subset=['Team'])
        advanced_df = advanced_df.rename(columns={
            'ORtg': 'Offensive_Rating',
            'DRtg': 'Defensive_Rating',
            'NRtg': 'Net_Rating',
            'Pace': 'Pace',
            'FTr': 'Free_Throw_Rate',
            'TS%': 'True_Shooting_Pct'
        })
        
        return games_df, advanced_df
    
    def fetch_player_stats(self, season):
        """Fetch player stats for a specific season"""
        url = f"{self.base_url}/leagues/NBA_{season}_per_game.html"
        players_df = pd.read_html(url)[0]
        
        # Clean player stats
        players_df = players_df[players_df['Rk'].notna()]  # Remove header rows
        players_df = players_df.rename(columns={
            'Player': 'Name',
            'MP': 'Minutes',
            'PTS': 'Points',
            'TRB': 'Rebounds',
            'AST': 'Assists'
        })
        
        return players_df

    def fetch_kaggle_data(self):
        """Fetch and process relevant Kaggle datasets"""
        # Make sure you've set up your Kaggle API credentials
        kaggle.api.authenticate()
        
        # List of relevant datasets
        datasets = [
            'nathanlauga/nba-games',  # Comprehensive game data
            'wyattowalsh/basketball',  # Advanced stats
        ]
        
        for dataset in datasets:
            print(f"Downloading {dataset}")
            kaggle.api.dataset_download_files(dataset, path='data', unzip=True)
        
        # Process downloaded data
        games_df = pd.read_csv('data/games.csv')
        games_details_df = pd.read_csv('data/games_details.csv')
        
        return self.process_kaggle_data(games_df, games_details_df)
    
    def process_kaggle_data(self, games_df, games_details_df):
        """Process Kaggle datasets"""
        # Merge game details with games
        df = pd.merge(
            games_df,
            games_details_df,
            on='GAME_ID',
            how='left'
        )
        
        # Calculate advanced metrics
        df['Efficiency'] = (
            df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] -
            (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TO']
        )
        
        return df
    
    def combine_data_sources(self, bref_games, bref_advanced, kaggle_data):
        """Combine data from both sources"""
        # Merge based on common fields (date, teams)
        # Add logic to combine the datasets based on your needs
        pass
    
    def save_data(self, games_df, advanced_df, players_df, output_dir='data'):
        """Save processed data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        games_df.to_csv(f'{output_dir}/nba_games.csv', index=False)
        advanced_df.to_csv(f'{output_dir}/nba_advanced.csv', index=False)
        players_df.to_csv(f'{output_dir}/nba_players.csv', index=False)
        
        print(f"Data saved to {output_dir}/")

    def fetch_standings(self):
        """Fetch standings data for each season"""
        all_standings = []
        
        for season in self.seasons:
            print(f"Fetching standings for season {season}-{season+1}")
            
            standings_url = f"{self.base_url}/leagues/NBA_{season}_standings.html"
            try:
                standings_df = pd.read_html(standings_url)[0]
                standings_df['Season'] = season
                standings_df['Date'] = pd.Timestamp(f"{season}-12-31")  # Approximate mid-season date
                
                # Clean standings data
                standings_df = standings_df.rename(columns={
                    'Team': 'team_name',
                    'W': 'wins',
                    'L': 'losses',
                    'GB': 'games_behind',
                    'PS/G': 'points_per_game',
                    'PA/G': 'points_allowed_per_game'
                })
                
                all_standings.append(standings_df)
                time.sleep(3)  # Be nice to the server
                
            except Exception as e:
                logging.error(f"Error fetching standings for {season}: {e}")
                continue
        
        return pd.concat(all_standings)

def main():
    collector = NBADataCollector()
    
    # Fetch Basketball Reference data
    print("Fetching Basketball Reference data...")
    games_df, advanced_df = collector.fetch_basketball_reference()
    
    # Fetch player stats for current season
    print("Fetching player stats...")
    players_df = collector.fetch_player_stats(2024)
    
    # Optional: Fetch Kaggle data
    try:
        print("Fetching Kaggle data...")
        kaggle_data = collector.fetch_kaggle_data()
    except Exception as e:
        print(f"Error fetching Kaggle data: {e}")
        kaggle_data = None
    
    # Save all data
    collector.save_data(games_df, advanced_df, players_df)
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()