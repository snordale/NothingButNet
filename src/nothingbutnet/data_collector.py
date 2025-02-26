import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import os
import logging
from io import StringIO
import random
import json
from pathlib import Path
from .database import get_session, save_game, save_team_stats, init_db

# Lazy import of kaggle to prevent automatic authentication
KAGGLE_AVAILABLE = False
try:
    import importlib
    kaggle_spec = importlib.util.find_spec("kaggle")
    KAGGLE_AVAILABLE = kaggle_spec is not None
except ImportError:
    pass

class NBADataCollector:
    def __init__(self):
        self.bref_url = "https://www.basketball-reference.com"
        self.espn_url = "https://www.espn.com/nba"
        self.seasons = range(2015, 2024)
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up sessions for different sources
        self.bref_session = self._create_session()
        self.espn_session = self._create_session()
        
        # Track request timestamps for rate limiting
        self.last_bref_request = 0
        self.last_espn_request = 0
        
    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session
    
    def _respect_rate_limit(self, source='bref'):
        """Ensure we respect rate limits for different sources"""
        if source == 'bref':
            # Basketball Reference: max 20 requests per minute
            min_delay = 6.0  # 6 seconds between requests to be safe
            last_request = self.last_bref_request
            self.last_bref_request = time.time()
        else:
            # ESPN: more lenient rate limiting
            min_delay = 3.0
            last_request = self.last_espn_request
            self.last_espn_request = time.time()
            
        elapsed = time.time() - last_request
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
    
    def _make_request(self, url, source='bref', max_retries=5, initial_delay=3):
        """Make a request with rate limiting and exponential backoff"""
        delay = initial_delay
        session = self.bref_session if source == 'bref' else self.espn_session
        
        for attempt in range(max_retries):
            try:
                self._respect_rate_limit(source)
                
                # Add jitter to delay
                jitter = random.uniform(0, 1)
                time.sleep(jitter)
                
                response = session.get(url)
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries - 1:
                        delay *= 2  # Exponential backoff
                        logging.warning(f"Rate limited by {source}. Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay *= 2
                    logging.warning(f"Request failed: {e}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                raise
                
        raise Exception(f"Failed to fetch {url} after {max_retries} attempts")
    
    def _get_cached_data(self, key, max_age_hours=24):
        """Get data from cache if it exists and is fresh"""
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            mtime = cache_file.stat().st_mtime
            age_hours = (time.time() - mtime) / 3600
            
            if age_hours < max_age_hours:
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass
        return None
    
    def _save_to_cache(self, data, key):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{key}.parquet"
        data.to_parquet(cache_file)
    
    def fetch_basketball_reference(self):
        """Fetch data from Basketball Reference with caching"""
        # Try cache first
        games_df = self._get_cached_data('bref_games')
        advanced_df = self._get_cached_data('bref_advanced')
        
        if games_df is not None and advanced_df is not None:
            logging.info("Using cached Basketball Reference data")
            return games_df, advanced_df
        
        all_games = []
        all_advanced = []
        
        for season in self.seasons:
            print(f"Fetching season {season}-{season+1} from Basketball Reference")
            
            try:
                # Get schedule/results
                schedule_url = f"{self.bref_url}/leagues/NBA_{season}_games.html"
                schedule_response = self._make_request(schedule_url, source='bref')
                schedule_df = pd.read_html(StringIO(schedule_response.text))[0]
                
                # Get team advanced stats
                advanced_url = f"{self.bref_url}/leagues/NBA_{season}_ratings.html"
                advanced_response = self._make_request(advanced_url, source='bref')
                advanced_df = pd.read_html(StringIO(advanced_response.text))[0]
                advanced_df['Season'] = season
                
                all_games.append(schedule_df)
                all_advanced.append(advanced_df)
                
            except Exception as e:
                logging.error(f"Error fetching Basketball Reference data for season {season}: {e}")
                # Try ESPN as backup
                try:
                    espn_data = self.fetch_espn_data(season)
                    if espn_data:
                        all_games.append(espn_data)
                except Exception as e2:
                    logging.error(f"Error fetching ESPN data for season {season}: {e2}")
                continue
        
        if not all_games or not all_advanced:
            raise ValueError("Failed to collect any data")
            
        games_df = pd.concat(all_games)
        advanced_df = pd.concat(all_advanced)
        
        # Process the data
        games_df, advanced_df = self.process_basketball_reference_data(games_df, advanced_df)
        
        # Cache the processed data
        self._save_to_cache(games_df, 'bref_games')
        self._save_to_cache(advanced_df, 'bref_advanced')
        
        return games_df, advanced_df
    
    def fetch_espn_data(self, season):
        """Fetch game data from ESPN as a backup source"""
        season_year = str(season)
        url = f"{self.espn_url}/schedule/_/season/{season_year}/seasontype/2"  # Regular season
        
        try:
            response = self._make_request(url, source='espn')
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse ESPN's schedule page
            # Note: Implementation will depend on ESPN's HTML structure
            # This is a placeholder for the actual implementation
            games = []
            for game in soup.find_all('div', class_='game-info'):
                games.append({
                    'date': None,  # Parse date
                    'away_team': None,  # Parse away team
                    'home_team': None,  # Parse home team
                    'away_points': None,  # Parse away points
                    'home_points': None  # Parse home points
                })
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logging.error(f"Error fetching ESPN data: {e}")
            return None
    
    def process_basketball_reference_data(self, games_df, advanced_df):
        """Clean and process Basketball Reference data"""
        session = get_session()
        
        # Clean games data
        games_df = games_df.rename(columns={
            'Visitor/Neutral': 'away_team',
            'Home/Neutral': 'home_team',
            'PTS.1': 'home_points',
            'PTS': 'away_points',
        })
        
        # Convert date
        games_df['date'] = pd.to_datetime(games_df['Date'])
        games_df = games_df.drop('Date', axis=1)
        
        # Handle MultiIndex columns in advanced stats
        advanced_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in advanced_df.columns]
        
        # Clean advanced stats
        column_mapping = {
            'Unnamed: 1_level_0_Team': 'team',
            'Unadjusted_ORtg': 'offensive_rating',
            'Unadjusted_DRtg': 'defensive_rating',
            'Unadjusted_NRtg': 'net_rating',
            'Adjusted_MOV/A': 'adjusted_mov',
            'Adjusted_ORtg/A': 'adjusted_ortg',
            'Adjusted_DRtg/A': 'adjusted_drtg',
            'Adjusted_NRtg/A': 'adjusted_nrtg'
        }
        
        # Only rename columns that exist
        rename_cols = {k: v for k, v in column_mapping.items() if k in advanced_df.columns}
        advanced_df = advanced_df.rename(columns=rename_cols)
        
        # Clean team names
        if 'team' in advanced_df.columns:
            advanced_df['team'] = advanced_df['team'].astype(str).replace(r'\*.*$', '', regex=True).str.strip()
        
        # Save games to database
        for _, row in games_df.iterrows():
            save_game(
                session=session,
                date=row['date'],
                season=row['Season'] if 'Season' in row else row['date'].year,
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=row['home_points'],
                away_score=row['away_points'],
                source='basketball_reference'
            )
        
        # Save team stats to database
        for _, row in advanced_df.iterrows():
            stats_dict = {
                'offensive_rating': row.get('offensive_rating'),
                'defensive_rating': row.get('defensive_rating'),
                'net_rating': row.get('net_rating'),
                'pace': row.get('pace'),
                'games_played': row.get('G'),
                'wins': row.get('W'),
                'losses': row.get('L')
            }
            
            save_team_stats(
                session=session,
                team_name=row['team'],
                date=datetime.now(),  # Use current date for stats snapshot
                season=row['Season'],
                stats_dict=stats_dict
            )
        
        session.close()
        return games_df, advanced_df
    
    def fetch_player_stats(self, season):
        """Fetch player stats for a specific season"""
        url = f"{self.bref_url}/leagues/NBA_{season}_per_game.html"
        response = self._make_request(url)
        players_df = pd.read_html(StringIO(response.text))[0]
        
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
        if not KAGGLE_AVAILABLE:
            logging.warning("Kaggle package not available. Skipping Kaggle data.")
            return None
            
        try:
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
        except Exception as e:
            logging.warning(f"Error fetching Kaggle data: {e}")
            return None
    
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
            
            standings_url = f"{self.bref_url}/leagues/NBA_{season}_standings.html"
            try:
                # Get the page content
                response = self._make_request(standings_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different table IDs (format changed over the years)
                standings_df = None
                
                # Try conference tables first (newer format)
                east_table = soup.find('table', {'id': 'confs_standings_E'})
                west_table = soup.find('table', {'id': 'confs_standings_W'})
                
                if east_table is not None and west_table is not None:
                    # Read both tables
                    east_df = pd.read_html(StringIO(str(east_table)))[0]
                    west_df = pd.read_html(StringIO(str(west_table)))[0]
                    
                    # Add conference column
                    east_df['conference'] = 'Eastern'
                    west_df['conference'] = 'Western'
                    
                    # Combine conferences
                    standings_df = pd.concat([east_df, west_df])
                    
                    # Rename team column based on conference
                    if 'Eastern Conference' in standings_df.columns:
                        standings_df['team'] = standings_df['Eastern Conference'].combine_first(standings_df['Western Conference'])
                        standings_df = standings_df.drop(['Eastern Conference', 'Western Conference'], axis=1)
                else:
                    # Try expanded standings (older format)
                    expanded_table = soup.find('table', {'id': 'standings'})
                    if expanded_table is not None:
                        standings_df = pd.read_html(StringIO(str(expanded_table)))[0]
                        if 'Team' in standings_df.columns:
                            standings_df = standings_df.rename(columns={'Team': 'team'})
                        elif 'Unnamed: 0' in standings_df.columns:
                            standings_df = standings_df.rename(columns={'Unnamed: 0': 'team'})
                
                if standings_df is None:
                    raise ValueError("Could not find standings table")
                
                # Add season and date
                standings_df['season'] = season
                standings_df['date'] = pd.Timestamp(f"{season}-12-31")  # Approximate mid-season date
                
                # Clean standings data
                column_mapping = {
                    'W': 'wins',
                    'L': 'losses',
                    'W/L%': 'win_pct',
                    'GB': 'games_behind',
                    'PS/G': 'points_per_game',
                    'PA/G': 'points_allowed_per_game',
                    'SRS': 'srs',
                    'ORtg': 'offensive_rating',
                    'DRtg': 'defensive_rating',
                    'NRtg': 'net_rating',
                    'MOV': 'margin_of_victory',
                    'Pace': 'pace'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in standings_df.columns}
                standings_df = standings_df.rename(columns=rename_cols)
                
                # Clean team names
                if 'team' in standings_df.columns:
                    standings_df['team'] = standings_df['team'].astype(str).replace(r'\*.*$', '', regex=True).str.strip()
                    standings_df['team_id'] = standings_df['team']
                
                # Drop rows that don't contain team data (division headers, etc.)
                standings_df = standings_df[standings_df['team'].notna()]
                standings_df = standings_df[standings_df['wins'].notna()]
                
                all_standings.append(standings_df)
                
            except Exception as e:
                logging.error(f"Error fetching standings for {season}: {e}")
                continue
        
        if not all_standings:
            raise ValueError("No standings data could be collected")
            
        standings_df = pd.concat(all_standings, ignore_index=True)
        
        # Drop any unnecessary columns
        standings_df = standings_df.loc[:, ~standings_df.columns.str.contains('^Unnamed')]
        
        return standings_df

def main():
    # Initialize database
    init_db()
    
    collector = NBADataCollector()
    
    # Fetch Basketball Reference data
    print("Fetching Basketball Reference data...")
    games_df, advanced_df = collector.fetch_basketball_reference()
    
    # Fetch player stats for current season
    print("Fetching player stats...")
    players_df = collector.fetch_player_stats(2024)
    
    # Optional: Fetch Kaggle data
    try:
        print("Skipping Kaggle data...")
        # kaggle_data = collector.fetch_kaggle_data()
    except Exception as e:
        print(f"Error fetching Kaggle data: {e}")
        kaggle_data = None
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()