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
    def __init__(self):
        self.data_collector = NBADataCollector()
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
        games_df, advanced_df = self.data_collector.fetch_basketball_reference()
        players_df = self.data_collector.fetch_player_stats(2024)
        standings_df = self.data_collector.fetch_standings()
        
        logger.info("Training model...")
        history = self.predictor.train(games_df, players_df, standings_df)
        
        logger.info(f"Training complete! Test MAE: {history['test_mae']:.1f} points")
        self.predictor.save_model(self.model_path)
    
    def get_predictions(self, target_date):
        """Get predictions for games on target date"""
        logger.info(f"Getting predictions for {target_date}")
        
        # Fetch latest data
        games_df, advanced_df = self.data_collector.fetch_basketball_reference()
        players_df = self.data_collector.fetch_player_stats(2024)
        standings_df = self.data_collector.fetch_standings()
        
        # Get games for target date
        target_games = games_df[games_df['Date'].dt.date == target_date]
        
        if target_games.empty:
            logger.info(f"No games found for {target_date}")
            return []
        
        predictions = []
        for _, game in target_games.iterrows():
            try:
                # Prepare features for this game
                features = self.predictor.prepare_game_features(
                    games_df[games_df['Date'] < game['Date']],
                    players_df,
                    standings_df,
                    game
                )
                
                # Get model prediction
                predicted_spread = self.predictor.predict_spread(features)
                
                # Calculate confidence based on model uncertainty and historical performance
                confidence = self._calculate_confidence(
                    predicted_spread,
                    game['Spread'],
                    features
                )
                
                # Determine key factors
                key_factors = self._identify_key_factors(
                    game,
                    features,
                    predicted_spread,
                    games_df
                )
                
                prediction = {
                    'date': game['Date'].date(),
                    'home_team': game['Home'],
                    'away_team': game['Away'],
                    'spread': game['Spread'],
                    'predicted_spread': float(predicted_spread),
                    'confidence': float(confidence),
                    'key_factors': key_factors,
                    'bet_recommendation': self._get_bet_recommendation(
                        predicted_spread,
                        game['Spread'],
                        confidence
                    )
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting game {game['Home']} vs {game['Away']}: {e}")
                continue
        
        # Save predictions for later analysis
        if predictions:
            self.performance_tracker.save_prediction(predictions, target_date)
        
        return predictions
    
    def _calculate_confidence(self, predicted_spread, actual_spread, features):
        """Calculate confidence score for prediction"""
        # Base confidence on historical performance with similar predictions
        base_confidence = 0.5
        
        # Adjust based on prediction magnitude
        spread_diff = abs(predicted_spread - actual_spread)
        if spread_diff > 10:
            base_confidence += 0.2
        elif spread_diff > 5:
            base_confidence += 0.1
        
        # Adjust based on data quality
        if features.get('missing_data_ratio', 0) > 0.2:
            base_confidence -= 0.1
        
        # Cap confidence
        return min(0.95, max(0.1, base_confidence))
    
    def _identify_key_factors(self, game, features, predicted_spread, historical_games):
        """Identify key factors influencing the prediction"""
        factors = []
        
        # Recent form
        home_recent = historical_games[
            (historical_games['Home'] == game['Home']) &
            (historical_games['Date'] < game['Date'])
        ].tail(10)
        
        away_recent = historical_games[
            (historical_games['Away'] == game['Away']) &
            (historical_games['Date'] < game['Date'])
        ].tail(10)
        
        # Check recent ATS performance
        home_ats_record = (home_recent['Home_Points'] - home_recent['Away_Points'] > home_recent['Spread']).mean()
        away_ats_record = (away_recent['Away_Points'] - away_recent['Home_Points'] > -away_recent['Spread']).mean()
        
        if home_ats_record > 0.6:
            factors.append(f"{game['Home']} strong ATS ({home_ats_record:.0%})")
        if away_ats_record > 0.6:
            factors.append(f"{game['Away']} strong ATS ({away_ats_record:.0%})")
        
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
    
    def _get_bet_recommendation(self, predicted_spread, actual_spread, confidence):
        """Generate betting recommendation"""
        min_confidence = CONFIG['prediction']['min_confidence_for_bet']
        
        if confidence < min_confidence:
            return {
                'bet': False,
                'reason': "Confidence too low",
                'size': 0
            }
        
        # Determine bet size based on confidence and edge
        edge = abs(predicted_spread - actual_spread)
        if edge < 2:
            return {
                'bet': False,
                'reason': "Insufficient edge",
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
            'side': 'HOME' if predicted_spread > actual_spread else 'AWAY',
            'size': bet_size,
            'reason': f"Edge: {edge:.1f} points, Confidence: {confidence:.0%}"
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
        output.append(f"Current Spread: {pred['spread']:+.1f} ({pred['home_team']})")
        output.append(f"Predicted Spread: {pred['predicted_spread']:+.1f}")
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