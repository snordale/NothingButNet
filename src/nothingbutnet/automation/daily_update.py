#!/usr/bin/env python3
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
from python_crontab import CronTab

from ..config import CONFIG
from ..data_collector import NBADataCollector
from ..models.spread_predictor import SpreadPredictor
from ..performance_tracker import PerformanceTracker
from ..predict import GamePredictor

logger = logging.getLogger(__name__)

class AutomationManager:
    def __init__(self):
        self.data_collector = NBADataCollector()
        self.predictor = GamePredictor()
        self.performance_tracker = PerformanceTracker()
        
        # Set up logging
        log_dir = Path(CONFIG['paths']['logs']) / 'automation'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'automation.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_cron_jobs(self):
        """Set up automated tasks"""
        cron = CronTab(user=True)
        
        # Clear existing jobs
        cron.remove_all(comment='nothingbutnet')
        
        # Daily data update and predictions
        update_time = CONFIG['automation']['update_time']
        hour, minute = map(int, update_time.split(':'))
        
        update_job = cron.new(
            command=f'cd {Path.cwd()} && python3 -m nothingbutnet.automation.daily_update',
            comment='nothingbutnet'
        )
        update_job.hour.on(hour)
        update_job.minute.on(minute)
        
        # Weekly model retraining
        retrain_job = cron.new(
            command=f'cd {Path.cwd()} && python3 -m nothingbutnet.automation.daily_update --retrain',
            comment='nothingbutnet'
        )
        retrain_job.dow.on(6)  # Saturday
        retrain_job.hour.on(1)  # 1 AM
        
        cron.write()
        logger.info("Cron jobs set up successfully")
    
    def update(self):
        """Run daily update tasks"""
        logger.info("Starting daily update")
        
        try:
            # Fetch new data
            logger.info("Fetching new data...")
            games_df, advanced_df = self.data_collector.fetch_basketball_reference()
            players_df = self.data_collector.fetch_player_stats(2024)
            standings_df = self.data_collector.fetch_standings(force_refresh=True)  # Force refresh standings
            
            # Save raw data
            logger.info("Saving raw data...")
            games_df.to_parquet(Path(CONFIG['paths']['raw_data']) / 'games.parquet')
            advanced_df.to_parquet(Path(CONFIG['paths']['raw_data']) / 'advanced.parquet')
            players_df.to_parquet(Path(CONFIG['paths']['raw_data']) / 'players.parquet')
            standings_df.to_parquet(Path(CONFIG['paths']['raw_data']) / 'standings.parquet')
            
            # Update performance tracking
            self._update_performance_tracking(games_df)
            
            # Generate predictions for today
            today = datetime.now().date()
            predictions = self.predictor.get_predictions(today)
            
            if predictions:
                logger.info(f"Generated {len(predictions)} predictions for {today}")
            else:
                logger.info(f"No games found for {today}")
            
            # Analyze performance and generate insights
            self._analyze_and_report()
            
        except Exception as e:
            logger.error(f"Error in daily update: {e}", exc_info=True)
            raise
    
    def retrain_model(self):
        """Retrain the model with latest data"""
        logger.info("Starting model retraining")
        
        try:
            # Get latest performance analysis
            analysis = self.performance_tracker.get_latest_analysis()
            
            if analysis and analysis['overall']['accuracy_ats'] > 0.53:
                logger.info("Current model performing well, skipping retraining")
                return
            
            # Train new model
            self.predictor._train_new_model()
            
            logger.info("Model retraining complete")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}", exc_info=True)
            raise
    
    def _update_performance_tracking(self, games_df):
        """Update performance tracking with actual results"""
        yesterday = datetime.now().date() - timedelta(days=1)
        pred_file = Path(CONFIG['paths']['predictions']) / f"prediction_{yesterday.strftime('%Y%m%d')}.json"
        
        if not pred_file.exists():
            return
        
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        
        # Get actual results
        yesterday_games = games_df[games_df['Date'].dt.date == yesterday]
        
        for pred in predictions:
            game = yesterday_games[
                (yesterday_games['Home'] == pred['home_team']) &
                (yesterday_games['Away'] == pred['away_team'])
            ]
            
            if not game.empty:
                actual_result = {
                    'final_spread': float(game.iloc[0]['Home_Points'] - game.iloc[0]['Away_Points']),
                    'home_score': int(game.iloc[0]['Home_Points']),
                    'away_score': int(game.iloc[0]['Away_Points'])
                }
                pred['actual_result'] = actual_result
        
        # Save updated predictions
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
    
    def _analyze_and_report(self):
        """Analyze performance and generate report"""
        analysis = self.performance_tracker.analyze_performance()
        if not analysis:
            return
        
        insights = self.performance_tracker.generate_insights()
        
        # Generate report
        report = [
            "Daily Performance Report",
            "=" * 50,
            f"\nOverall Performance:",
            f"ATS Accuracy: {analysis['overall']['accuracy_ats']:.1%}",
            f"ROI: {analysis['overall']['roi']:.1%}",
            f"Average Margin Error: {analysis['overall']['average_margin_error']:.1f} points",
            
            "\nPerformance by Confidence Level:",
        ]
        
        for level, stats in analysis['by_confidence'].items():
            report.append(
                f"{level}: {stats['correct_ats']['mean']:.1%} "
                f"({stats['correct_ats']['count']} bets)"
            )
        
        if insights:
            report.append("\nKey Insights:")
            for insight in insights:
                prefix = "⚠️ " if insight['type'] == 'warning' else "💡 "
                report.append(f"{prefix}{insight['message']}")
                report.append(f"   Suggestion: {insight['suggestion']}")
        
        # Save report
        report_path = Path(CONFIG['paths']['logs']) / 'reports'
        report_path.mkdir(exist_ok=True)
        
        with open(report_path / f"report_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
            f.write("\n".join(report))

def main():
    manager = AutomationManager()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', action='store_true', help='Set up cron jobs')
    parser.add_argument('--retrain', action='store_true', help='Retrain model')
    args = parser.parse_args()
    
    if args.setup:
        manager.setup_cron_jobs()
    elif args.retrain:
        manager.retrain_model()
    else:
        manager.update()

if __name__ == "__main__":
    main() 