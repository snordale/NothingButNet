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
from .database import get_session, save_game, save_team_stats, init_db, Game, Team, TeamStats, upsert_team
from sqlalchemy import func, and_
from .config import CONFIG

# Lazy import of kaggle to prevent automatic authentication
KAGGLE_AVAILABLE = False
try:
    import importlib
    kaggle_spec = importlib.util.find_spec("kaggle")
    KAGGLE_AVAILABLE = kaggle_spec is not None
except ImportError:
    pass

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
        """Fetch data from Basketball Reference with incremental updates"""
        session = get_session()
        logging.info("Starting Basketball Reference data collection")
        
        try:
            # Get existing games by season
            existing_games = pd.read_sql(
                'SELECT season, COUNT(*) as game_count FROM games GROUP BY season',
                session.bind
            ).set_index('season')['game_count'].to_dict()
            
            logging.info("Existing games by season:")
            for season, count in existing_games.items():
                logging.info(f"  {season}: {count} games")
            
            # Determine seasons to fetch
            current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            seasons_to_fetch = []
            
            # Expected games per season (accounting for COVID seasons)
            expected_games = {
                2021: 1080,  # COVID-shortened (72 games per team)
                2020: 1080,  # COVID-interrupted
            }
            # All other seasons should have 1230 games (30 teams × 82 games ÷ 2)
            
            # Check each season in our range
            for season in self.seasons:
                games_in_season = existing_games.get(season, 0)
                expected = expected_games.get(season, 1230)
                
                # Add season if:
                # 1. It's the current season
                # 2. It's missing games
                # 3. We have no games from it
                if (season == current_season or 
                    games_in_season < expected or 
                    season not in existing_games):
                    seasons_to_fetch.append(season)
            
            if not seasons_to_fetch:
                logging.info("All seasons have expected number of games")
                return None, None
            
            logging.info(f"Will fetch data for seasons: {seasons_to_fetch}")
            
            all_games = []
            all_advanced = []
            
            # Track progress
            total_games = 0
            successful_games = 0
            failed_games = 0
            
            for season in seasons_to_fetch:
                logging.info(f"Processing season {season}")
                months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may']
                season_games = []
                
                for month in months:
                    try:
                        schedule_url = f"{self.bref_url}/leagues/NBA_{season}_games-{month}.html"
                        logging.info(f"Fetching {month} {season} games from {schedule_url}")
                        
                        schedule_response = self._make_request(schedule_url, source='bref', max_retries=3)
                        if schedule_response and schedule_response.ok:
                            schedule_df = pd.read_html(StringIO(schedule_response.text))[0]
                            
                            # Convert date before filtering
                            schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
                            
                            # Don't filter by date anymore - we'll handle duplicates later
                            schedule_df['Season'] = season
                            season_games.append(schedule_df)
                            games_count = len(schedule_df)
                            total_games += games_count
                            successful_games += games_count
                            logging.info(f"Added {games_count} games from {month} {season}")
                                
                    except Exception as e:
                        failed_games += 1
                        logging.warning(f"Error fetching {month} {season} games: {e}")
                        continue
                    
                    # Save progress after each month
                    if season_games:
                        try:
                            season_df = pd.concat(season_games, ignore_index=True)
                            
                            # Remove duplicates based on date and teams
                            season_df = season_df.drop_duplicates(
                                subset=['Date', 'Visitor/Neutral', 'Home/Neutral'],
                                keep='first'
                            )
                            
                            # Process and save to database immediately
                            processed_games, processed_advanced = self.process_basketball_reference_data(
                                season_df, 
                                pd.DataFrame()  # Empty advanced stats, will fetch separately
                            )
                            
                            if not processed_games.empty:
                                all_games.append(processed_games)
                                logging.info(f"Saved {len(processed_games)} games to database")
                                
                        except Exception as e:
                            logging.error(f"Error processing {season} {month} data: {e}")
                    
                # Fetch advanced stats for the season
                try:
                    advanced_url = f"{self.bref_url}/leagues/NBA_{season}_ratings.html"
                    logging.info(f"Fetching advanced stats for season {season}")
                    
                    advanced_response = self._make_request(advanced_url, source='bref')
                    if advanced_response and advanced_response.ok:
                        advanced_df = pd.read_html(StringIO(advanced_response.text))[0]
                        advanced_df['Season'] = season
                        all_advanced.append(advanced_df)
                        logging.info(f"Added advanced stats for season {season}")
                        
                except Exception as e:
                    logging.error(f"Error fetching advanced stats for season {season}: {e}")
                
                # Respect rate limits between seasons
                time.sleep(3)
            
            # Final stats
            logging.info("Data collection complete!")
            logging.info(f"Total games processed: {total_games}")
            logging.info(f"Successful: {successful_games}")
            logging.info(f"Failed: {failed_games}")
            
            if not all_games:
                logging.warning("No new games collected")
                return None, None
                
            games_df = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
            advanced_df = pd.concat(all_advanced, ignore_index=True) if all_advanced else pd.DataFrame()
            
            # Verify final game counts
            if not games_df.empty:
                final_counts = games_df.groupby('Season').size()
                logging.info("\nFinal game counts by season:")
                for season, count in final_counts.items():
                    expected = expected_games.get(season, 1230)
                    logging.info(f"  {season}: {count} games (Expected: {expected})")
            
            return games_df, advanced_df
            
        except Exception as e:
            logging.error(f"Error in data collection: {e}")
            return None, None
        finally:
            session.close()
    
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
        
        # Filter out playoff games before date conversion
        games_df = games_df[games_df['Date'] != 'Playoffs']
        
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
        
        # Add team IDs by creating/getting teams from database
        team_ids = {}
        for team_name in set(games_df['home_team'].unique()) | set(games_df['away_team'].unique()):
            if pd.notna(team_name):
                team = upsert_team(session, team_name)
                team_ids[team_name] = team.id
        
        # Add team IDs to games DataFrame
        games_df['home_team_id'] = games_df['home_team'].map(team_ids)
        games_df['away_team_id'] = games_df['away_team'].map(team_ids)
        
        # Add point differential and winner columns
        games_df['point_differential'] = games_df['home_points'] - games_df['away_points']
        games_df['home_team_won'] = games_df['point_differential'] > 0
        
        # Drop unnecessary columns
        cols_to_drop = ['Unnamed: 6', 'Unnamed: 7', 'LOG', 'Arena', 'Notes']
        games_df = games_df.drop([col for col in cols_to_drop if col in games_df.columns], axis=1)
        
        # Handle MultiIndex columns in advanced stats
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
            'Adjusted_NRtg/A': 'adjusted_nrtg'
        }
        
        # Only rename columns that exist
        rename_cols = {k: v for k, v in column_mapping.items() if k in advanced_df.columns}
        advanced_df = advanced_df.rename(columns=rename_cols)
        
        # Clean team names
        if 'team' in advanced_df.columns:
            advanced_df['team'] = advanced_df['team'].astype(str).replace(r'\*.*$', '', regex=True).str.strip()
            # Add team IDs to advanced stats
            advanced_df['team_id'] = advanced_df['team'].map(team_ids)
        
        # Save games to database
        for _, row in games_df.iterrows():
            try:
                # Convert NaN scores to None for database insertion
                home_score = None if pd.isna(row['home_points']) else int(row['home_points'])
                away_score = None if pd.isna(row['away_points']) else int(row['away_points'])
                
                save_game(
                    session=session,
                    date=row['date'],
                    season=row['Season'] if 'Season' in row else row['date'].year,
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    home_score=home_score,
                    away_score=away_score,
                    source='basketball_reference'
                )
            except Exception as e:
                logging.error(f"Error saving game {row['home_team']} vs {row['away_team']}: {e}")
                continue
        
        # Save team stats to database
        for _, row in advanced_df.iterrows():
            try:
                stats_dict = {
                    'offensive_rating': row.get('offensive_rating'),
                    'defensive_rating': row.get('defensive_rating'),
                    'net_rating': row.get('net_rating'),
                    'pace': row.get('pace'),
                    'games_played': row.get('G'),
                    'wins': row.get('W'),
                    'losses': row.get('L')
                }
                
                # Convert any NaN values to None for database insertion
                stats_dict = {k: None if pd.isna(v) else v for k, v in stats_dict.items()}
                
                save_team_stats(
                    session=session,
                    team_name=row['team'],
                    date=datetime.now(),  # Use current date for stats snapshot
                    season=row['Season'],
                    stats_dict=stats_dict
                )
            except Exception as e:
                logging.error(f"Error saving stats for team {row['team']}: {e}")
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