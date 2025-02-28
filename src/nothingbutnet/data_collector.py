import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import os
from io import StringIO
import random
import json
from pathlib import Path
from .database import get_session, save_game, save_team_stats, init_db, Game, Team, TeamStats, upsert_team
from sqlalchemy import func, and_
from .config import CONFIG

def setup_kaggle():
    try:
        import kaggle
        return True
    except ImportError:
        logging.warning("Kaggle package not available. Installing kaggle package...")
        try:
            import subprocess
            subprocess.check_call(['pip3', 'install', 'kaggle'])
            import kaggle
            return True
        except Exception as e:
            logging.error(f"Error installing kaggle package: {e}")
            return False

KAGGLE_AVAILABLE = setup_kaggle()

class NBADataCollector:
    def __init__(self, reload_historical=False, cache_ttl_hours=24):
        """Initialize the data collector"""
        self.reload_historical = reload_historical
        self.cache_ttl_hours = cache_ttl_hours
        
        # URLs
        self.bref_url = "https://www.basketball-reference.com"
        self.espn_url = "https://www.espn.com/nba"
        
        # Rate limiting
        self.last_request = {}  # Track last request time for each source
        
        # Create session for requests
        self.session = requests.Session()
        
        # Get seasons to collect (last 10 seasons by default)
        current_year = datetime.now().year
        self.seasons = list(range(current_year - 9, current_year + 1))
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Team name mappings
        self.team_name_map = {
            # ESPN to Basketball Reference
            'LA Clippers': 'Los Angeles Clippers',
            'LA Lakers': 'Los Angeles Lakers',
            'NY Knicks': 'New York Knicks',
            'GS Warriors': 'Golden State Warriors',
            'SA Spurs': 'San Antonio Spurs',
            'UTAH': 'Utah Jazz',
            'PHX': 'Phoenix Suns',
            'SAC': 'Sacramento Kings',
            'POR': 'Portland Trail Blazers',
            'OKC': 'Oklahoma City Thunder',
            'NO Pelicans': 'New Orleans Pelicans',
            'MIN': 'Minnesota Timberwolves',
            'MEM': 'Memphis Grizzlies',
            'HOU': 'Houston Rockets',
            'DEN': 'Denver Nuggets',
            'DAL': 'Dallas Mavericks',
            'WAS': 'Washington Wizards',
            'TOR': 'Toronto Raptors',
            'PHI': 'Philadelphia 76ers',
            'ORL': 'Orlando Magic',
            'MIL': 'Milwaukee Bucks',
            'MIA': 'Miami Heat',
            'IND': 'Indiana Pacers',
            'DET': 'Detroit Pistons',
            'CLE': 'Cleveland Cavaliers',
            'CHI': 'Chicago Bulls',
            'CHA': 'Charlotte Hornets',
            'BOS': 'Boston Celtics',
            'BKN': 'Brooklyn Nets',
            'ATL': 'Atlanta Hawks'
        }
        
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
        """Respect rate limits for different data sources"""
        current_time = time.time()
        
        if source == 'bref':
            # Basketball Reference: max 20 requests per minute
            # Use 6s delay to be safe (10 requests per minute)
            required_delay = 6
        elif source == 'espn':
            required_delay = 3
        else:
            required_delay = 3
        
        # Check if we need to wait
        if source in self.last_request:
            time_since_last = current_time - self.last_request[source]
            if time_since_last < required_delay:
                time.sleep(required_delay - time_since_last)
        
        # Update last request time
        self.last_request[source] = time.time()
    
    def _make_request(self, url, source='bref', max_retries=3, initial_delay=3):
        """Make an HTTP request with rate limiting and retries"""
        
        # Respect rate limits
        self._respect_rate_limit(source)
        
        # Add jitter to avoid synchronized requests
        jitter = random.uniform(0.1, 1.0)
        time.sleep(jitter)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Exponential backoff for retries
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logging.info(f"Successfully fetched {url}")
                    return response
                    
                elif response.status_code == 404:
                    logging.info(f"No data found at {url} (404)")
                    return None
                    
                elif response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', delay * 2))
                    logging.warning(f"Rate limited on attempt {attempt + 1}, waiting {retry_after}s")
                    time.sleep(retry_after)
                    delay *= 2  # Double the delay for next attempt
                    
                else:
                    logging.warning(f"Request failed with status {response.status_code} on attempt {attempt + 1}")
                    time.sleep(delay)
                    delay *= 2  # Double the delay for next attempt
                    
            except requests.Timeout:
                logging.warning(f"Request timed out on attempt {attempt + 1}")
                time.sleep(delay)
                delay *= 2
                
            except requests.RequestException as e:
                logging.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
                time.sleep(delay)
                delay *= 2
        
        logging.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None
    
    def _get_cached_data(self, key, max_age_hours=24):
        """Get data from cache if it exists and is fresh"""
        cache_file = self.cache_dir / f"{key}.parquet"
        
        if not cache_file.exists():
            logging.debug(f"Cache miss: {key} (file does not exist)")
            return None
            
        try:
            mtime = cache_file.stat().st_mtime
            age_hours = (time.time() - mtime) / 3600
            
            # Check if cache is stale
            if age_hours >= max_age_hours:
                logging.debug(f"Cache miss: {key} (age: {age_hours:.1f} hours > {max_age_hours} hours)")
                return None
            
            # Load cached data
            data = pd.read_parquet(cache_file)
            
            # Validate cache data
            if data.empty:
                logging.warning(f"Empty cache data for {key}")
                return None
            
            logging.debug(f"Cache hit: {key} (age: {age_hours:.1f} hours)")
            return data
            
        except Exception as e:
            logging.warning(f"Error reading cache for {key}: {str(e)}")
            
            try:
                # Remove corrupted cache file
                cache_file.unlink()
                logging.info(f"Removed corrupted cache file for {key}")
            except Exception as e:
                logging.error(f"Error removing corrupted cache file for {key}: {str(e)}")
            
            return None
    
    def _save_to_cache(self, data, key):
        """Save data to cache with validation"""
        if data is None or data.empty:
            logging.warning(f"Attempted to cache empty data for {key}")
            return False
            
        cache_file = self.cache_dir / f"{key}.parquet"
        temp_file = cache_file.with_suffix('.tmp')
        
        try:
            # Convert string columns that should be numeric to appropriate types
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Try to convert attendance and other numeric columns
                    if col == 'Attend.':
                        data[col] = pd.to_numeric(
                            data[col].astype(str).str.replace(',', ''),
                            errors='coerce'
                        )
                    else:
                        # Try to convert to numeric if it looks like a number
                        try:
                            numeric_data = pd.to_numeric(data[col], errors='coerce')
                            if numeric_data.notna().any():
                                data[col] = numeric_data
                        except:
                            pass
            
            # Save to temporary file first
            data.to_parquet(temp_file)
            
            # Validate the saved data by reading it back
            test_read = pd.read_parquet(temp_file)
            if test_read.empty:
                raise ValueError("Saved cache file is empty")
            
            # If validation passes, move to final location
            temp_file.replace(cache_file)
            logging.debug(f"Successfully cached data for {key}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching data for {key}: {str(e)}")
            
            # Clean up temporary file
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
                
            return False
    
    def fetch_basketball_reference(self):
        """Fetch data from Basketball Reference"""
        print("Fetching Basketball Reference data...")
        
        games_df = pd.DataFrame()
        advanced_df = pd.DataFrame()
        
        try:
            # Get current season
            current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            
            # Fetch data for each season
            for season in range(current_season - 9, current_season + 1):
                print(f"\nProcessing season {season}...")
                
                # Fetch monthly schedule pages
                for month in ['october', 'november', 'december', 'january', 'february', 'march', 'april']:
                    try:
                        # Construct URL
                        if month in ['october', 'november', 'december']:
                            year = season
                        else:
                            year = season + 1
                            
                        url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html'
                        
                        # Check cache first
                        cache_key = f'bref_schedule_{season}_{month}'
                        cached_data = self._get_cached_data(cache_key)
                        
                        if cached_data is not None:
                            month_df = cached_data
                        else:
                            # Respect rate limit
                            self._respect_rate_limit('bref')
                            
                            # Fetch data
                            response = self._make_request(url)
                            if response is None:
                                continue
                                
                            # Parse HTML
                            tables = pd.read_html(response.text)
                            if not tables:
                                continue
                                
                            month_df = tables[0]
                            month_df['Season'] = season
                            
                            # Cache the data
                            self._save_to_cache(month_df, cache_key)
                        
                        # Append to main DataFrame
                        games_df = pd.concat([games_df, month_df], ignore_index=True)
                        
                    except Exception as e:
                        logging.warning(f"Error fetching {month} {season} games: {e}")
                        continue
                
                # Fetch team stats
                try:
                    stats_url = f'https://www.basketball-reference.com/leagues/NBA_{season}.html'
                    cache_key = f'bref_stats_{season}'
                    cached_stats = self._get_cached_data(cache_key)
                    
                    if cached_stats is not None:
                        season_stats = cached_stats
                    else:
                        # Respect rate limit
                        self._respect_rate_limit('bref')
                        
                        # Fetch data
                        response = self._make_request(stats_url)
                        if response is None:
                            continue
                            
                        # Parse HTML
                        tables = pd.read_html(response.text)
                        if len(tables) < 4:  # We need the advanced stats table
                            continue
                            
                        season_stats = tables[3]  # Advanced stats table
                        season_stats['Season'] = season
                        
                        # Cache the data
                        self._save_to_cache(season_stats, cache_key)
                    
                    print(f"Found stats for {season}:")
                    print(season_stats.columns.tolist())
                    print(f"First row: {season_stats.iloc[0].to_dict()}")
                    
                    # Append to main DataFrame
                    advanced_df = pd.concat([advanced_df, season_stats], ignore_index=True)
                    
                except Exception as e:
                    logging.error(f"Error fetching team stats for season {season}: {e}")
                    continue
        
        except Exception as e:
            logging.error(f"Error in data collection: {e}")
            raise
        
        # Process the data
        if not games_df.empty and not advanced_df.empty:
            return self.process_basketball_reference_data(games_df, advanced_df)
        else:
            logging.error("No data collected")
            return pd.DataFrame(), pd.DataFrame()
    
    def fetch_espn_data(self, season):
        """Fetch game data from ESPN as a backup source"""
        season_year = str(season)
        url = f"{self.espn_url}/schedule/_/season/{season_year}/seasontype/2"  # Regular season
        
        try:
            response = self._make_request(url, source='espn')
            soup = BeautifulSoup(response.text, 'html.parser')
            
            games = []
            # Find all game rows
            game_rows = soup.find_all('div', {'class': 'ScheduleTables'})
            
            for date_section in game_rows:
                try:
                    # Get date from section header
                    date_str = date_section.find('h2', {'class': 'Table__Title'}).text.strip()
                    game_date = pd.to_datetime(date_str)
                    
                    # Find all games for this date
                    game_items = date_section.find_all('tr', {'class': 'Table__TR'})
                    
                    for game in game_items:
                        try:
                            teams = game.find_all('td', {'class': 'Table__TD'})
                            if len(teams) < 2:
                                continue
                                
                            away_team = teams[0].find('a', {'class': 'AnchorLink'}).text.strip()
                            home_team = teams[1].find('a', {'class': 'AnchorLink'}).text.strip()
                            
                            # Standardize team names
                            away_team = self.standardize_team_name(away_team)
                            home_team = self.standardize_team_name(home_team)
                            
                            # Try to get scores if game is complete
                            score_cells = game.find_all('td', {'class': 'Table__TD'})
                            away_score = None
                            home_score = None
                            
                            if len(score_cells) >= 4:
                                try:
                                    away_score = int(score_cells[2].text.strip())
                                    home_score = int(score_cells[3].text.strip())
                                except (ValueError, IndexError):
                                    pass
                            
                            games.append({
                                'Date': game_date,
                                'Season': season,
                                'away_team': away_team,
                                'home_team': home_team,
                                'away_points': away_score,
                                'home_points': home_score,
                                'source': 'espn'
                            })
                            
                        except Exception as e:
                            logging.warning(f"Error parsing game: {e}")
                            continue
                            
                except Exception as e:
                    logging.warning(f"Error parsing date section: {e}")
                    continue
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logging.error(f"Error fetching ESPN data: {e}")
            return None
    
    def process_basketball_reference_data(self, games_df, advanced_df):
        """Clean and process Basketball Reference data"""
        session = get_session()
        
        # Filter out playoff games and preseason games before date conversion
        games_df = games_df[~games_df['Date'].isin(['Playoffs', 'Preseason'])]
        
        # Clean games data
        games_df = games_df.rename(columns={
            'Visitor/Neutral': 'away_team',
            'Home/Neutral': 'home_team',
            'PTS.1': 'home_points',
            'PTS': 'away_points',
            'Start (ET)': 'start_time'
        })
        
        # Convert points to numeric, replacing any non-numeric values with NaN
        games_df['home_points'] = pd.to_numeric(games_df['home_points'], errors='coerce')
        games_df['away_points'] = pd.to_numeric(games_df['away_points'], errors='coerce')
        
        # Convert date
        games_df['date'] = pd.to_datetime(games_df['Date'])
        games_df = games_df.drop('Date', axis=1)
        
        # Filter out preseason games based on date (preseason games are typically in October before regular season start)
        def determine_game_type(row):
            game_date = row['date']
            season = row['Season']
            
            # Regular season typically starts mid-to-late October
            season_start = pd.Timestamp(f"{season}-10-15")
            season_end = pd.Timestamp(f"{season+1}-04-15")
            
            # In-season tournament started in 2023
            if season >= 2023 and 'Notes' in row and isinstance(row['Notes'], str):
                if 'IST' in row['Notes'] or 'In-Season Tournament' in row['Notes']:
                    return 'in_season_tournament'
            
            # Check if it's a playoff game (after regular season end)
            if game_date > season_end:
                return 'playoffs'
            
            # Filter out preseason games (before regular season start)
            if game_date < season_start:
                return 'preseason'
            
            return 'regular_season'
        
        # Add game type
        games_df['game_type'] = games_df.apply(determine_game_type, axis=1)
        
        # Filter out preseason games
        games_df = games_df[games_df['game_type'] != 'preseason']
        
        # Process advanced stats
        if isinstance(advanced_df.columns, pd.MultiIndex):
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
            'Adjusted_NRtg/A': 'adjusted_nrtg',
            'W': 'wins',
            'L': 'losses',
            'W/L%': 'win_pct',
            'GB': 'games_behind',
            'PS/G': 'points_per_game',
            'PA/G': 'points_allowed_per_game'
        }
        
        # Only rename columns that exist
        rename_cols = {k: v for k, v in column_mapping.items() if k in advanced_df.columns}
        advanced_df = advanced_df.rename(columns=rename_cols)
        
        # Clean team names and add team IDs
        if 'team' in advanced_df.columns:
            advanced_df['team'] = advanced_df['team'].astype(str).replace(r'\*.*$', '', regex=True).str.strip()
            
            # Calculate games played
            advanced_df['games_played'] = advanced_df['wins'] + advanced_df['losses']
            
            # Save team stats to database
            for _, row in advanced_df.iterrows():
                try:
                    team = upsert_team(session, row['team'])
                    
                    # Create stats dict with all available columns
                    stats_dict = {
                        'offensive_rating': row.get('offensive_rating'),
                        'defensive_rating': row.get('defensive_rating'),
                        'net_rating': row.get('net_rating'),
                        'pace': row.get('pace'),
                        'games_played': row.get('games_played'),
                        'wins': row.get('wins'),
                        'losses': row.get('losses'),
                        'win_pct': row.get('win_pct'),
                        'points_per_game': row.get('points_per_game'),
                        'points_allowed_per_game': row.get('points_allowed_per_game'),
                        'games_behind': row.get('games_behind', 0)
                    }
                    
                    # Save stats for each date in the season
                    for date in games_df['date'].unique():
                        save_team_stats(
                            session=session,
                            team_name=row['team'],
                            date=date,
                            season=games_df['Season'].iloc[0],  # Use first season value
                            stats_dict=stats_dict
                        )
                except Exception as e:
                    logging.error(f"Error saving stats for team {row['team']}: {e}")
                    continue
        
        # Save games to database
        for _, row in games_df.iterrows():
            try:
                # Convert NaN scores to None for database insertion
                home_score = None if pd.isna(row['home_points']) else int(row['home_points'])
                away_score = None if pd.isna(row['away_points']) else int(row['away_points'])
                
                save_game(
                    session=session,
                    date=row['date'],
                    season=row['Season'],
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    home_score=home_score,
                    away_score=away_score,
                    game_type=row['game_type'],
                    source='basketball_reference'
                )
            except Exception as e:
                logging.error(f"Error saving game {row['home_team']} vs {row['away_team']}: {e}")
                continue
        
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
            logging.warning("Kaggle package not available. Installing kaggle package...")
            try:
                import subprocess
                subprocess.check_call(['pip3', 'install', 'kaggle'])
                import kaggle
                global KAGGLE_AVAILABLE
                KAGGLE_AVAILABLE = True
            except Exception as e:
                logging.error(f"Error installing kaggle package: {e}")
                return None
            
        try:
            # Make sure you've set up your Kaggle API credentials
            import kaggle
            kaggle.api.authenticate()
            
            # Download NBA games dataset
            dataset = 'wyattowalsh/basketball'
            print(f"Downloading {dataset}")
            
            # Create data directory if it doesn't exist
            import os
            os.makedirs('data/kaggle', exist_ok=True)
            
            # Download and unzip the dataset
            kaggle.api.dataset_download_files(dataset, path='data/kaggle', unzip=True)
            
            # Read the relevant CSV files
            games_df = pd.read_csv('data/kaggle/games.csv')
            games_details_df = pd.read_csv('data/kaggle/games_details.csv')
            teams_df = pd.read_csv('data/kaggle/teams.csv')
            
            # Process the data
            processed_data = self.process_kaggle_data(games_df, games_details_df, teams_df)
            
            return processed_data
        except Exception as e:
            logging.error(f"Error fetching Kaggle data: {e}")
            return None
    
    def process_kaggle_data(self, games_df, games_details_df, teams_df):
        """Process Kaggle datasets"""
        session = get_session()
        
        try:
            # Clean and process games data
            games_df['date'] = pd.to_datetime(games_df['GAME_DATE_EST'])
            games_df['season'] = games_df['SEASON']
            
            # Map team IDs to names using teams_df
            team_id_to_name = dict(zip(teams_df['TEAM_ID'], teams_df['NICKNAME']))
            
            games_df['home_team'] = games_df['HOME_TEAM_ID'].map(team_id_to_name)
            games_df['away_team'] = games_df['VISITOR_TEAM_ID'].map(team_id_to_name)
            games_df['home_points'] = games_df['PTS_home']
            games_df['away_points'] = games_df['PTS_away']
            
            # Determine game type
            def determine_game_type(row):
                if row['GAME_STATUS_TEXT'] == 'PPD':  # Postponed games
                    return None
                elif row['SEASON_TYPE'] == 'Regular Season':
                    return 'regular_season'
                elif row['SEASON_TYPE'] == 'Playoffs':
                    return 'playoffs'
                elif row['SEASON_TYPE'] == 'Pre Season':
                    return 'preseason'
                else:
                    return 'regular_season'  # Default to regular season
            
            games_df['game_type'] = games_df.apply(determine_game_type, axis=1)
            
            # Filter out games without a type (postponed games)
            games_df = games_df[games_df['game_type'].notna()]
            
            # Save teams to database
            for _, team in teams_df.iterrows():
                try:
                    upsert_team(
                        session=session,
                        name=team['NICKNAME'],
                        abbreviation=team['ABBREVIATION'],
                        conference=team['CONFERENCE'],
                        division=team['DIVISION']
                    )
                except Exception as e:
                    logging.error(f"Error saving team {team['NICKNAME']}: {e}")
                    continue
            
            # Save games to database
            for _, game in games_df.iterrows():
                try:
                    save_game(
                        session=session,
                        date=game['date'],
                        season=game['season'],
                        home_team=game['home_team'],
                        away_team=game['away_team'],
                        home_score=game['home_points'],
                        away_score=game['away_points'],
                        game_type=game['game_type'],
                        source='kaggle'
                    )
                except Exception as e:
                    logging.error(f"Error saving game {game['home_team']} vs {game['away_team']}: {e}")
                    continue
            
            # Calculate and save team stats
            for season in games_df['season'].unique():
                season_games = games_df[games_df['season'] == season]
                
                for team_id in teams_df['TEAM_ID'].unique():
                    team_name = team_id_to_name[team_id]
                    
                    # Get home and away games for the team
                    home_games = season_games[season_games['HOME_TEAM_ID'] == team_id]
                    away_games = season_games[season_games['VISITOR_TEAM_ID'] == team_id]
                    
                    # Calculate basic stats
                    wins = len(home_games[home_games['home_points'] > home_games['away_points']]) + \
                           len(away_games[away_games['away_points'] > away_games['home_points']])
                           
                    losses = len(home_games[home_games['home_points'] < home_games['away_points']]) + \
                            len(away_games[away_games['away_points'] < away_games['home_points']])
                            
                    games_played = len(home_games) + len(away_games)
                    
                    if games_played > 0:
                        points_scored = home_games['home_points'].sum() + away_games['away_points'].sum()
                        points_allowed = home_games['away_points'].sum() + away_games['home_points'].sum()
                        
                        stats_dict = {
                            'games_played': games_played,
                            'wins': wins,
                            'losses': losses,
                            'win_pct': wins / games_played if games_played > 0 else 0,
                            'points_per_game': points_scored / games_played,
                            'points_allowed_per_game': points_allowed / games_played,
                            'offensive_rating': (points_scored / games_played) * 100,  # Simple offensive rating
                            'defensive_rating': (points_allowed / games_played) * 100,  # Simple defensive rating
                            'net_rating': ((points_scored - points_allowed) / games_played) * 100,
                            'pace': 100.0  # Default pace
                        }
                        
                        # Save stats for the last date of each month in the season
                        monthly_dates = pd.date_range(
                            start=season_games['date'].min(),
                            end=season_games['date'].max(),
                            freq='M'
                        )
                        
                        for date in monthly_dates:
                            save_team_stats(
                                session=session,
                                team_name=team_name,
                                date=date,
                                season=season,
                                stats_dict=stats_dict
                            )
            
            session.commit()
            return games_df
            
        except Exception as e:
            logging.error(f"Error processing Kaggle data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
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

    def fetch_standings(self, force_refresh=False):
        """Fetch current standings, using database first, then cache, then remote"""
        if not force_refresh:
            # Try database first
            session = get_session()
            try:
                # Get latest standings for each team
                latest_stats = (
                    session.query(TeamStats)
                    .from_self(
                        TeamStats.team_id,
                        TeamStats.date,
                        func.max(TeamStats.date).label('max_date')
                    )
                    .group_by(TeamStats.team_id)
                    .subquery()
                )
                
                stats = (
                    session.query(TeamStats)
                    .join(
                        latest_stats,
                        and_(
                            TeamStats.team_id == latest_stats.c.team_id,
                            TeamStats.date == latest_stats.c.max_date
                        )
                    )
                    .all()
                )
                
                if stats:
                    # Convert to DataFrame
                    data = []
                    for stat in stats:
                        data.append({
                            'team_id': stat.team_id,
                            'team': stat.team.name,
                            'date': stat.date,
                            'season': stat.season,
                            'wins': stat.wins,
                            'losses': stat.losses,
                            'position': stat.position,
                            'games_behind': stat.games_behind,
                            'conference': stat.conference,
                            'win_pct': stat.win_pct,
                            'points_per_game': stat.points_per_game,
                            'points_allowed_per_game': stat.points_allowed_per_game,
                            'offensive_rating': stat.offensive_rating,
                            'defensive_rating': stat.defensive_rating,
                            'net_rating': stat.net_rating,
                            'pace': stat.pace
                        })
                    
                    standings_df = pd.DataFrame(data)
                    session.close()
                    return standings_df
            except Exception as e:
                logging.warning(f"Error fetching standings from database: {e}")
                session.close()
            
            # Try cache next
            cache_file = Path('data/cache/standings.parquet')
            if cache_file.exists():
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < self.cache_ttl_hours * 3600:
                    try:
                        return pd.read_parquet(cache_file)
                    except Exception as e:
                        logging.warning(f"Error reading cached standings: {e}")
        
        # If we get here, we need to fetch fresh data
        all_standings = []
        
        for season in range(self.start_season, self.end_season + 1):
            print(f"Fetching standings for season {season}-{season+1}")
            
            # For current season, use the standings page
            if season == self.end_season:
                standings_url = f"{self.bref_url}/leagues/NBA_{season}_standings.html"
            else:
                standings_url = f"{self.bref_url}/leagues/NBA_{season}.html"
            
            try:
                # Get the page content
                response = self._make_request(standings_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different table IDs (format changed over the years)
                standings_df = None
                
                # Try Eastern Conference table
                for east_id in ['confs_standings_E', 'divs_standings_E', 'e_standings']:
                    east_table = soup.find('table', {'id': east_id})
                    if east_table is not None:
                        break
                
                # Try Western Conference table
                for west_id in ['confs_standings_W', 'divs_standings_W', 'w_standings']:
                    west_table = soup.find('table', {'id': west_id})
                    if west_table is not None:
                        break
                
                if east_table is not None and west_table is not None:
                    # Read both tables
                    east_df = pd.read_html(StringIO(str(east_table)))[0]
                    west_df = pd.read_html(StringIO(str(west_table)))[0]
                    
                    # Add conference column
                    east_df['conference'] = 'Eastern'
                    west_df['conference'] = 'Western'
                    
                    # Combine conferences
                    standings_df = pd.concat([east_df, west_df])
                    
                else:
                    # Try expanded standings (older format)
                    for table_id in ['expanded_standings', 'standings']:
                        table = soup.find('table', {'id': table_id})
                        if table is not None:
                            standings_df = pd.read_html(StringIO(str(table)))[0]
                            break
                
                if standings_df is None:
                    logging.warning(f"Could not find standings table for season {season}")
                    continue
                
                # Clean up the team names
                if 'Team' in standings_df.columns:
                    standings_df['team'] = standings_df['Team']
                elif 'Eastern Conference' in standings_df.columns:
                    standings_df['team'] = standings_df['Eastern Conference'].combine_first(standings_df['Western Conference'])
                elif 'Unnamed: 0' in standings_df.columns:
                    standings_df['team'] = standings_df['Unnamed: 0']
                elif 'Eastern Division' in standings_df.columns:
                    standings_df['team'] = standings_df['Eastern Division'].combine_first(standings_df['Western Division'])
                
                # Clean team names and remove asterisks and other markers
                standings_df['team'] = standings_df['team'].astype(str).replace({
                    r'\*.*$': '',  # Remove asterisk and everything after
                    r'\(\d+\)': '',  # Remove seeding numbers in parentheses
                    r'^\s+|\s+$': ''  # Remove leading/trailing whitespace
                }, regex=True)
                
                # Add season and date
                standings_df['season'] = season
                standings_df['date'] = pd.Timestamp(f"{season}-12-31")  # Mid-season date
                
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
                    'Pace': 'pace',
                    'Rk': 'rank'
                }
                
                # Only rename columns that exist
                rename_cols = {k: v for k, v in column_mapping.items() if k in standings_df.columns}
                standings_df = standings_df.rename(columns=rename_cols)
                
                # Calculate position within conference if not already present
                if 'conference' in standings_df.columns:
                    standings_df['position'] = standings_df.groupby('conference')['wins'].rank(ascending=False, method='min')
                else:
                    standings_df['position'] = standings_df['wins'].rank(ascending=False, method='min')
                
                # Calculate games behind if not present
                if 'games_behind' not in standings_df.columns:
                    max_wins = standings_df['wins'].max()
                    min_losses = standings_df['losses'].min()
                    standings_df['games_behind'] = ((max_wins - standings_df['wins']) + (standings_df['losses'] - min_losses)) / 2.0
                
                # Drop rows that don't contain team data (division headers, etc.)
                standings_df = standings_df[standings_df['team'].notna()]
                standings_df = standings_df[standings_df['wins'].notna()]
                
                # Convert numeric columns
                numeric_columns = ['wins', 'losses', 'win_pct', 'games_behind', 'points_per_game', 
                                 'points_allowed_per_game', 'srs', 'offensive_rating', 'defensive_rating', 
                                 'net_rating', 'margin_of_victory', 'pace']
                
                for col in numeric_columns:
                    if col in standings_df.columns:
                        standings_df[col] = pd.to_numeric(standings_df[col], errors='coerce')
                
                all_standings.append(standings_df)
                
                # Respect rate limits
                time.sleep(3)
                
            except Exception as e:
                logging.error(f"Error fetching standings for {season}: {e}")
                continue
        
        if not all_standings:
            logging.error("No standings data could be collected")
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
            
        standings_df = pd.concat(all_standings, ignore_index=True)
        standings_df = standings_df.loc[:, ~standings_df.columns.str.contains('^Unnamed')]
        
        # Save to database
        session = get_session()
        try:
            for _, row in standings_df.iterrows():
                stats_dict = row.to_dict()
                save_team_stats(
                    session=session,
                    team_name=row['team'],
                    date=row['date'],
                    season=row['season'],
                    stats_dict=stats_dict
                )
        except Exception as e:
            logging.warning(f"Error saving standings to database: {e}")
        finally:
            session.close()
        
        # Cache the results
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            standings_df.to_parquet(cache_file)
        except Exception as e:
            logging.warning(f"Error caching standings: {e}")
        
        return standings_df

    def standardize_team_name(self, name):
        """Convert team names to Basketball Reference format"""
        return self.team_name_map.get(name, name)

def main():
    # Initialize database
    init_db()
    
    collector = NBADataCollector()
    
    # Try Kaggle data first
    print("Fetching Kaggle data...")
    try:
        kaggle_data = collector.fetch_kaggle_data()
        if kaggle_data is not None:
            print("Successfully loaded Kaggle data")
            return
    except Exception as e:
        print(f"Error fetching Kaggle data: {e}")
    
    # Fall back to Basketball Reference if Kaggle fails
    print("Falling back to Basketball Reference data...")
    games_df, advanced_df = collector.fetch_basketball_reference()
    
    # Fetch player stats for current season
    print("Fetching player stats...")
    players_df = collector.fetch_player_stats(2024)
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()