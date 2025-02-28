import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Create data directory if it doesn't exist
data_dir = Path(__file__).parent.parent.parent / 'data'
data_dir.mkdir(exist_ok=True)

# Use SQLite database
DATABASE_URL = f'sqlite:///{data_dir}/nba.db'

# Create base class for declarative models
Base = declarative_base()

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String)
    conference = Column(String)
    division = Column(String)

class Game(Base):
    __tablename__ = 'games'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    game_type = Column(String)  # regular_season, playoffs, preseason
    
    # Additional columns for betting and game context
    point_spread = Column(Float)  # Positive means home team favored
    total_line = Column(Float)    # Over/under
    rest_days_home = Column(Integer)
    rest_days_away = Column(Integer)
    
    # Relationships for easier querying
    home_team = relationship('Team', foreign_keys=[home_team_id])
    away_team = relationship('Team', foreign_keys=[away_team_id])

class TeamStats(Base):
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    
    # Basic stats
    games_played = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    points_per_game = Column(Float)
    points_allowed_per_game = Column(Float)
    win_pct = Column(Float)
    
    # Additional stats for model features
    last_10_wins = Column(Integer)
    last_10_losses = Column(Integer)
    last_10_points_scored = Column(Float)
    last_10_points_allowed = Column(Float)
    home_wins = Column(Integer)
    home_losses = Column(Integer)
    away_wins = Column(Integer)
    away_losses = Column(Integer)
    streak = Column(Integer)  # Positive for win streak, negative for loss streak
    
    # Advanced stats
    point_diff = Column(Float)  # Average point differential
    home_point_diff = Column(Float)
    away_point_diff = Column(Float)
    last_10_point_diff = Column(Float)
    
    # New advanced stats
    offensive_rating = Column(Float)  # Points scored per 100 possessions
    defensive_rating = Column(Float)  # Points allowed per 100 possessions
    net_rating = Column(Float)  # Offensive rating minus defensive rating
    pace = Column(Float)  # Possessions per 48 minutes
    
    # Rebounding stats
    offensive_rebounds_per_game = Column(Float)
    defensive_rebounds_per_game = Column(Float)
    total_rebounds_per_game = Column(Float)
    offensive_rebound_pct = Column(Float)  # Percentage of available offensive rebounds grabbed
    defensive_rebound_pct = Column(Float)  # Percentage of available defensive rebounds grabbed
    
    # Turnover stats
    turnovers_per_game = Column(Float)
    turnover_pct = Column(Float)  # Turnovers per 100 possessions
    
    # Relationship
    team = relationship('Team')

def init_db(drop_all=False):
    """Initialize the database"""
    try:
        engine = create_engine(DATABASE_URL)
        
        if drop_all:
            Base.metadata.drop_all(engine)
            
        Base.metadata.create_all(engine)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
        return Session()
        
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        raise

def get_session():
    """Get a new database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def upsert_team(session, name, abbreviation=None, conference=None, division=None):
    """Insert or update a team"""
    team = session.query(Team).filter_by(name=name).first()
    if team is None:
        team = Team(name=name, abbreviation=abbreviation,
                   conference=conference, division=division)
        session.add(team)
    else:
        team.abbreviation = abbreviation or team.abbreviation
        team.conference = conference or team.conference
        team.division = division or team.division
    
    session.commit()
    return team

def save_game(session, date, season, home_team, away_team, home_score=None, away_score=None, 
              game_type='regular_season', point_spread=None, total_line=None,
              rest_days_home=None, rest_days_away=None):
    """Save a game to the database"""
    # Get team IDs
    home_team_id = upsert_team(session, home_team).id
    away_team_id = upsert_team(session, away_team).id
    
    # Check if game already exists
    existing_game = session.query(Game).filter(
        Game.date == date,
        Game.home_team_id == home_team_id,
        Game.away_team_id == away_team_id
    ).first()
    
    if existing_game:
        # Update existing game
        existing_game.season = season
        existing_game.home_score = home_score
        existing_game.away_score = away_score
        existing_game.game_type = game_type
        existing_game.point_spread = point_spread
        existing_game.total_line = total_line
        existing_game.rest_days_home = rest_days_home
        existing_game.rest_days_away = rest_days_away
        game = existing_game
    else:
        # Create new game
        game = Game(
            date=date,
            season=season,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_score=home_score,
            away_score=away_score,
            game_type=game_type,
            point_spread=point_spread,
            total_line=total_line,
            rest_days_home=rest_days_home,
            rest_days_away=rest_days_away
        )
        session.add(game)
    
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    
    return game

def save_team_stats(session, team_name, date, season, stats_dict):
    """Save team stats to the database"""
    team = upsert_team(session, team_name)
    
    stats = session.query(TeamStats).filter_by(
        team_id=team.id,
        date=date
    ).first()
    
    if stats is None:
        stats = TeamStats(team_id=team.id, date=date, season=season, **stats_dict)
        session.add(stats)
    else:
        for key, value in stats_dict.items():
            setattr(stats, key, value)
    
    session.commit()
    return stats 