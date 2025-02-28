#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import pandas as pd

from .data_collector import NBADataCollector
from .models.spread_predictor import SpreadPredictor
from .performance_tracker import PerformanceTracker
from .config import CONFIG

logger = logging.getLogger(__name__)

class GamePredictor:
    def __init__(self, reload_historical=False):
        self.data_collector = NBADataCollector(reload_historical=reload_historical)
        self.predictor = SpreadPredictor()
        self.performance_tracker = PerformanceTracker()
        self.model_path = Path(CONFIG['paths']['models']) / 'spread_predictor.pt'
        
        # Load model if it exists
        if self.model_path.exists():
            try:
                self.predictor.load_model(self.model_path)
                logger.info("Loaded existing model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._train_new_model()
        else:
            logger.info("No existing model found. Training new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new model with historical data"""
        logger.info("Fetching historical data...")
        try:
            # Force reload of historical data for training
            self.data_collector.reload_historical = True
            games_df, advanced_df = self.data_collector.fetch_basketball_reference()
            standings_df = self.data_collector.fetch_standings()
            
            if games_df.empty or advanced_df.empty:
                raise ValueError("No data available for training")
            
            logger.info("Training model...")
            history = self.predictor.train(games_df, standings_df)
            
            logger.info(f"Training complete! Test MAE: {history['test_mae']:.1f} points")
            self.predictor.save_model(self.model_path)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def get_predictions(self, target_date):
        """Get predictions for games on target date"""
        logger.info(f"Getting predictions for {target_date}")
        
        try:
            # Fetch historical data
            games_df, advanced_df = self.data_collector.fetch_basketball_reference()
            standings_df = self.data_collector.fetch_standings()
            
            if games_df.empty or advanced_df.empty:
                logger.error("No game data available")
                return []
            
            # Fetch upcoming games from ESPN
            current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            upcoming_games = self.data_collector.fetch_espn_data(current_season)
            
            if upcoming_games is None or upcoming_games.empty:
                logger.error("Could not fetch upcoming games")
                return []
            
            # Get games for target date
            target_games = upcoming_games[pd.to_datetime(upcoming_games['Date']).dt.date == target_date]
            
            if target_games.empty:
                logger.info(f"No games found for {target_date}")
                return []
            
            predictions = []
            for _, game in target_games.iterrows():
                try:
                    # Create a game record in the same format as historical games
                    game_record = pd.Series({
                        'date': game['Date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'home_team_id': None,  # Will be filled from team_ids
                        'away_team_id': None,
                        'home_points': None,
                        'away_points': None,
                        'Season': current_season
                    })
                    
                    # Get team IDs from historical data
                    home_team_id = games_df[games_df['home_team'] == game['home_team']]['home_team_id'].iloc[0]
                    away_team_id = games_df[games_df['away_team'] == game['away_team']]['away_team_id'].iloc[0]
                    game_record['home_team_id'] = home_team_id
                    game_record['away_team_id'] = away_team_id
                    
                    # Prepare features for this game
                    features = self.predictor.prepare_game_features(
                        games_df[pd.to_datetime(games_df['date']) < game['Date']],
                        standings_df,
                        game_record
                    )
                    
                    if features is None or features.empty:
                        logger.warning(f"Could not prepare features for {game['home_team']} vs {game['away_team']}")
                        continue
                    
                    # Get model prediction
                    predicted_spread = self.predictor.predict_spread(features)
                    
                    # Calculate confidence based on model uncertainty
                    confidence = self._calculate_confidence(
                        predicted_spread,
                        features
                    )
                    
                    # Determine key factors
                    key_factors = self._identify_key_factors(
                        game_record,
                        features,
                        predicted_spread,
                        games_df
                    )
                    
                    prediction = {
                        'date': game['Date'].date() if isinstance(game['Date'], pd.Timestamp) else game['Date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'predicted_spread': float(predicted_spread),
                        'confidence': float(confidence),
                        'key_factors': key_factors,
                        'bet_recommendation': self._get_bet_recommendation(
                            predicted_spread,
                            confidence
                        )
                    }
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Error predicting game {game['home_team']} vs {game['away_team']}: {e}")
                    continue
            
            # Save predictions for later analysis
            if predictions:
                self.performance_tracker.save_prediction(predictions, target_date)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            return []
    
    def _calculate_confidence(self, predicted_spread, features):
        """Calculate confidence score for prediction"""
        try:
            # Base confidence on data quality and prediction magnitude
            base_confidence = 0.5
            
            # Adjust based on prediction magnitude
            if abs(predicted_spread) > 10:
                base_confidence += 0.2
            elif abs(predicted_spread) > 5:
                base_confidence += 0.1
            
            # Adjust based on data quality
            missing_ratio = features.isna().mean().mean() if not features.empty else 0
            if missing_ratio > 0.2:
                base_confidence -= 0.1 * (missing_ratio / 0.2)
            
            # Cap confidence
            return min(0.95, max(0.1, base_confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5  # Return default confidence on error
    
    def _identify_key_factors(self, game, features, predicted_spread, historical_games):
        """Identify key factors influencing the prediction"""
        factors = []
        
        # Recent form
        home_recent = historical_games[
            (historical_games['home_team'] == game['home_team']) &
            (historical_games['date'] < game['date'])
        ].tail(10)
        
        away_recent = historical_games[
            (historical_games['away_team'] == game['away_team']) &
            (historical_games['date'] < game['date'])
        ].tail(10)
        
        # Check recent performance
        home_win_rate = (home_recent['home_points'] > home_recent['away_points']).mean()
        away_win_rate = (away_recent['away_points'] > away_recent['home_points']).mean()
        
        if home_win_rate > 0.7:
            factors.append(f"{game['home_team']} strong form ({home_win_rate:.0%} win rate)")
        if away_win_rate > 0.7:
            factors.append(f"{game['away_team']} strong form ({away_win_rate:.0%} win rate)")
        
        # Check rest advantage
        if features.get('home_rest_days', 0) > features.get('away_rest_days', 0) + 2:
            factors.append("Home rest advantage")
        elif features.get('away_rest_days', 0) > features.get('home_rest_days', 0) + 2:
            factors.append("Away rest advantage")
        
        # Check injuries
        if features.get('home_player_starter_availability', 1) < 0.8:
            factors.append("Home team missing starters")
        if features.get('away_player_starter_availability', 1) < 0.8:
            factors.append("Away team missing starters")
        
        return factors
    
    def _get_bet_recommendation(self, predicted_spread, confidence):
        """Generate betting recommendation"""
        min_confidence = CONFIG['prediction']['min_confidence_for_bet']
        
        if confidence < min_confidence:
            return {
                'bet': False,
                'reason': "Confidence too low",
                'size': 0
            }
        
        # Determine bet size based on confidence and edge
        if abs(predicted_spread) < 2:
            return {
                'bet': False,
                'reason': "Spread too small",
                'size': 0
            }
        
        # Kelly criterion for bet sizing
        win_prob = confidence
        odds = 0.909  # Standard -110 odds
        kelly = (win_prob * odds - (1 - win_prob)) / odds
        
        # Conservative Kelly (1/4)
        bet_size = max(0.25, min(1.0, kelly * 0.25))
        
        return {
            'bet': True,
            'side': 'HOME' if predicted_spread > 0 else 'AWAY',
            'size': bet_size,
            'reason': f"Edge: {abs(predicted_spread):.1f} points, Confidence: {confidence:.0%}"
        }

def format_predictions(predictions):
    """Format predictions for display"""
    if not predictions:
        return "No predictions available"
        
    output = []
    output.append(f"\nPredictions for {predictions[0]['date']}\n")
    output.append("=" * 80)
    
    for pred in predictions:
        output.append(f"\n{pred['away_team']} @ {pred['home_team']}")
        output.append(f"Current Spread: {pred['predicted_spread']:+.1f}")
        output.append(f"Confidence: {pred['confidence']:.0%}")
        
        if pred['key_factors']:
            output.append("Key Factors:")
            for factor in pred['key_factors']:
                output.append(f"  • {factor}")
        
        rec = pred['bet_recommendation']
        if rec['bet']:
            output.append(f"\nBET RECOMMENDATION: {rec['side']} ({rec['size']:.2f} units)")
            output.append(f"Reason: {rec['reason']}")
        else:
            output.append(f"\nNo bet recommended: {rec['reason']}")
        
        output.append("-" * 80)
    
    # Add performance summary if available
    tracker = PerformanceTracker()
    analysis = tracker.get_latest_analysis()
    if analysis:
        output.append("\nRecent Performance:")
        output.append(f"Overall ATS: {analysis['overall']['accuracy_ats']:.1%}")
        output.append(f"ROI: {analysis['overall']['roi']:.1%}")
        
        insights = tracker.generate_insights()
        if insights:
            output.append("\nInsights:")
            for insight in insights:
                if insight['type'] == 'warning':
                    output.append(f"⚠️  {insight['message']}")
                else:
                    output.append(f"💡 {insight['message']}")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Predict NBA game outcomes")
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.now().date(),
        help="Target date for predictions (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    
    predictor = GamePredictor()
    predictions = predictor.get_predictions(args.date)
    print(format_predictions(predictions))

if __name__ == "__main__":
    main() 