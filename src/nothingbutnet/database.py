import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import pandas as pd

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Get database URL from environment or use default
DATABASE_URL = os.getenv('NBA_DATABASE_URL', 'postgresql://localhost/nothingbutnet')

# Create base class for declarative models
Base = declarative_base()

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String)
    conference = Column(String)
    division = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    home_games = relationship('Game', foreign_keys='Game.home_team_id', back_populates='home_team')
    away_games = relationship('Game', foreign_keys='Game.away_team_id', back_populates='away_team')
    stats = relationship('TeamStats', back_populates='team')

class Game(Base):
    __tablename__ = 'games'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    spread = Column(Float)  # Closing spread (positive means home team favored)
    total = Column(Float)   # Over/under
    status = Column(String) # scheduled, in_progress, final
    source = Column(String) # basketball_reference, espn, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    home_team = relationship('Team', foreign_keys=[home_team_id], back_populates='home_games')
    away_team = relationship('Team', foreign_keys=[away_team_id], back_populates='away_games')
    predictions = relationship('Prediction', back_populates='game')

class TeamStats(Base):
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    games_played = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    net_rating = Column(Float)
    pace = Column(Float)
    position = Column(Integer)  # Added: Position in standings
    games_behind = Column(Float)  # Added: Games behind leader
    conference = Column(String)  # Added: Conference
    win_pct = Column(Float)  # Added: Win percentage
    points_per_game = Column(Float)  # Added: Points per game
    points_allowed_per_game = Column(Float)  # Added: Points allowed per game
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship('Team', back_populates='stats')

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), nullable=False)
    predicted_spread = Column(Float, nullable=False)  # Positive means home team favored
    confidence = Column(Float, nullable=False)
    model_version = Column(String)
    features_used = Column(String)  # JSON string of feature names
    created_at = Column(DateTime, default=datetime.utcnow)
    was_correct = Column(Boolean)  # Set after game completion
    
    # Relationships
    game = relationship('Game', back_populates='predictions')

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

def save_game(session, date, season, home_team, away_team, home_score=None,
              away_score=None, spread=None, total=None, status='scheduled', source=None):
    """Save a game to the database"""
    # Ensure teams exist
    home = upsert_team(session, home_team)
    away = upsert_team(session, away_team)
    
    # Check if game exists
    game = session.query(Game).filter_by(
        date=date,
        home_team_id=home.id,
        away_team_id=away.id
    ).first()
    
    if game is None:
        game = Game(
            date=date,
            season=season,
            home_team_id=home.id,
            away_team_id=away.id,
            home_score=home_score,
            away_score=away_score,
            spread=spread,
            total=total,
            status=status,
            source=source
        )
        session.add(game)
    else:
        game.home_score = home_score or game.home_score
        game.away_score = away_score or game.away_score
        game.spread = spread or game.spread
        game.total = total or game.total
        game.status = status
        game.source = source or game.source
    
    session.commit()
    return game

def save_team_stats(session, team_name, date, season, stats_dict):
    """Save team stats to the database"""
    team = upsert_team(session, team_name)
    
    stats = session.query(TeamStats).filter_by(
        team_id=team.id,
        date=date
    ).first()
    
    # Convert any NaN values to None for database insertion
    stats_dict = {k: None if pd.isna(v) else v for k, v in stats_dict.items()}
    
    # Add required fields if not present
    required_fields = ['offensive_rating', 'defensive_rating', 'net_rating', 'pace',
                      'games_played', 'wins', 'losses', 'position', 'games_behind',
                      'conference', 'win_pct', 'points_per_game', 'points_allowed_per_game']
    
    for field in required_fields:
        if field not in stats_dict:
            stats_dict[field] = None
    
    if stats is None:
        stats = TeamStats(team_id=team.id, date=date, season=season, **stats_dict)
        session.add(stats)
    else:
        for key, value in stats_dict.items():
            setattr(stats, key, value)
    
    session.commit()
    return stats

def save_prediction(session, game_id, predicted_spread, confidence,
                   model_version, features_used):
    """Save a prediction to the database"""
    prediction = Prediction(
        game_id=game_id,
        predicted_spread=predicted_spread,
        confidence=confidence,
        model_version=model_version,
        features_used=features_used
    )
    
    session.add(prediction)
    session.commit()
    return prediction 