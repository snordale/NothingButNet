import pandas as pd
import numpy as np
from datetime import timedelta

class RecentATSPredictor:
    def __init__(self, lookback_games=10, min_games=5):
        self.lookback_games = lookback_games
        self.min_games = min_games
        
    def predict(self, game, games_df, advanced_df, standings_df):
        """Predict spread based on recent ATS performance"""
        game_date = pd.to_datetime(game['Date'])
        home_team = game['Home']
        away_team = game['Away']
        
        # Get recent games for both teams
        home_recent = self._get_recent_games(games_df, home_team, game_date)
        away_recent = self._get_recent_games(games_df, away_team, game_date)
        
        if len(home_recent) < self.min_games or len(away_recent) < self.min_games:
            return None
            
        # Calculate ATS performance
        home_ats_record = self._calculate_ats_record(home_recent, home_team)
        away_ats_record = self._calculate_ats_record(away_recent, away_team)
        
        # Get current form metrics
        home_form = self._calculate_form_metrics(home_recent, home_team)
        away_form = self._calculate_form_metrics(away_recent, away_team)
        
        # Calculate prediction confidence
        prediction, confidence, factors = self._make_prediction(
            home_ats_record, away_ats_record,
            home_form, away_form
        )
        
        return {
            'date': game_date.date(),
            'home_team': home_team,
            'away_team': away_team,
            'spread': game.get('Spread', 0),
            'favorite': home_team if game.get('Spread', 0) < 0 else away_team,
            'pick': prediction,
            'confidence': confidence,
            'key_factors': factors
        }
        
    def _get_recent_games(self, games_df, team, target_date):
        """Get recent games for a team before target date"""
        team_games = games_df[
            ((games_df['Home'] == team) | (games_df['Away'] == team)) &
            (games_df['Date'] < target_date)
        ].sort_values('Date', ascending=False)
        
        return team_games.head(self.lookback_games)
        
    def _calculate_ats_record(self, games, team):
        """Calculate ATS record for recent games"""
        ats_wins = 0
        total_games = len(games)
        
        for _, game in games.iterrows():
            is_home = game['Home'] == team
            actual_margin = game['Point_Spread'] if is_home else -game['Point_Spread']
            spread = game.get('Spread', 0)  # Assuming spread is from home team perspective
            spread = spread if is_home else -spread
            
            if actual_margin > spread:
                ats_wins += 1
                
        return {
            'ats_win_pct': ats_wins / total_games if total_games > 0 else 0,
            'total_games': total_games
        }
        
    def _calculate_form_metrics(self, games, team):
        """Calculate additional form metrics"""
        metrics = {
            'avg_margin': 0,
            'avg_points': 0,
            'avg_points_allowed': 0,
            'home_games': 0
        }
        
        for _, game in games.iterrows():
            is_home = game['Home'] == team
            team_points = game['Home_Points'] if is_home else game['Away_Points']
            opp_points = game['Away_Points'] if is_home else game['Home_Points']
            margin = team_points - opp_points
            
            metrics['avg_margin'] += margin
            metrics['avg_points'] += team_points
            metrics['avg_points_allowed'] += opp_points
            metrics['home_games'] += 1 if is_home else 0
            
        total_games = len(games)
        if total_games > 0:
            metrics['avg_margin'] /= total_games
            metrics['avg_points'] /= total_games
            metrics['avg_points_allowed'] /= total_games
            metrics['home_ratio'] = metrics['home_games'] / total_games
            
        return metrics
        
    def _make_prediction(self, home_ats, away_ats, home_form, away_form):
        """Make prediction based on ATS records and form"""
        factors = []
        
        # Calculate base confidence from ATS records
        home_strength = home_ats['ats_win_pct'] * min(1, home_ats['total_games'] / self.lookback_games)
        away_strength = away_ats['ats_win_pct'] * min(1, away_ats['total_games'] / self.lookback_games)
        
        # Adjust for form
        home_form_score = (
            home_form['avg_margin'] * 0.3 +
            (home_form['avg_points'] - home_form['avg_points_allowed']) * 0.2
        ) / 20  # Normalize to roughly -1 to 1
        
        away_form_score = (
            away_form['avg_margin'] * 0.3 +
            (away_form['avg_points'] - away_form['avg_points_allowed']) * 0.2
        ) / 20
        
        # Calculate final confidence
        home_final = home_strength + home_form_score * 0.3
        away_final = away_strength + away_form_score * 0.3
        
        # Determine prediction
        if home_final > away_final:
            prediction = "HOME"
            confidence = (home_final - away_final) * 0.5 + 0.5  # Scale to 0.5-1.0
            if home_ats['ats_win_pct'] > 0.6:
                factors.append(f"Home team strong ATS ({home_ats['ats_win_pct']:.0%})")
            if home_form['avg_margin'] > 5:
                factors.append("Home team good form")
        else:
            prediction = "AWAY"
            confidence = (away_final - home_final) * 0.5 + 0.5
            if away_ats['ats_win_pct'] > 0.6:
                factors.append(f"Away team strong ATS ({away_ats['ats_win_pct']:.0%})")
            if away_form['avg_margin'] > 5:
                factors.append("Away team good form")
                
        # Add form-based factors
        if abs(home_form['avg_margin'] - away_form['avg_margin']) > 7:
            factors.append("Significant recent margin difference")
            
        return prediction, min(0.95, confidence), factors 