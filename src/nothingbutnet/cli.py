#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys

from nothingbutnet.data_collector import NBADataCollector
from nothingbutnet.models.ensemble_predictor import EnsemblePredictor
from nothingbutnet.predictors.recent_ats_predictor import RecentATSPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GamePredictor:
    def __init__(self):
        self.data_collector = NBADataCollector()
        self.predictors = self._initialize_predictors()
        self.ensemble = EnsemblePredictor(self.predictors)
        
    def _initialize_predictors(self):
        """Initialize all prediction models"""
        predictors = [
            RecentATSPredictor()  # Start with the core ATS predictor
        ]
        
        # Load any additional predictors based on successful hypotheses
        hypothesis_dir = Path(__file__).parent / "automation/hypotheses"
        if hypothesis_dir.exists():
            try:
                with open(hypothesis_dir / "successful_models.json", "r") as f:
                    successful_models = json.load(f)
                    for model_info in successful_models:
                        # Dynamic import and initialization of successful models
                        module = __import__(f"nothingbutnet.predictors.{model_info['module']}", fromlist=[model_info['class']])
                        predictor_class = getattr(module, model_info['class'])
                        predictors.append(predictor_class())
            except Exception as e:
                logger.warning(f"Could not load additional predictors: {e}")
        
        return predictors
    
    def get_predictions(self, target_date):
        """Get predictions for games on the target date"""
        # Fetch latest data
        logger.info("Updating data...")
        games_df, advanced_df = self.data_collector.fetch_basketball_reference()
        standings_df = self.data_collector.fetch_standings()
        
        # Get games for target date
        target_games = games_df[games_df['Date'].dt.date == target_date]
        
        if target_games.empty:
            logger.info(f"No games found for {target_date}")
            return []
        
        # Get predictions from ensemble
        predictions = self.ensemble.predict(
            target_games,
            advanced_df,
            standings_df,
            target_date
        )
        
        # Save predictions for tracking
        self._save_predictions(predictions, target_date)
        
        return predictions
    
    def _save_predictions(self, predictions, target_date):
        """Save predictions for performance tracking"""
        predictions_dir = Path(__file__).parent / "automation/predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        prediction_file = predictions_dir / f"predictions_{target_date}.json"
        with open(prediction_file, "w") as f:
            json.dump(predictions, f, indent=2, default=str)

def format_predictions(predictions):
    """Format predictions for display"""
    output = []
    output.append("\nPredictions for {}\n".format(predictions[0]['date']))
    output.append("-" * 80)
    
    for pred in predictions:
        confidence_str = "*" * int(pred['confidence'] * 5)  # Visual confidence indicator
        output.append(f"{pred['away_team']} @ {pred['home_team']}")
        output.append(f"Spread: {pred['spread']:+.1f} ({pred['favorite']})")
        output.append(f"Prediction: {pred['pick']} {confidence_str}")
        output.append(f"Confidence: {pred['confidence']:.2%}")
        output.append(f"Key Factors: {', '.join(pred['key_factors'])}")
        output.append("-" * 80)
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Predict NBA game outcomes against the spread")
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.now().date(),
        help="Target date for predictions (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    
    predictor = GamePredictor()
    predictions = predictor.get_predictions(args.date)
    
    if predictions:
        print(format_predictions(predictions))
    else:
        print(f"No predictions available for {args.date}")

if __name__ == "__main__":
    main() 