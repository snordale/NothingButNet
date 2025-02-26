import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from .config import CONFIG

logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.performance_dir = Path(CONFIG['paths']['performance'])
        self.predictions_dir = Path(CONFIG['paths']['predictions'])
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
    def save_prediction(self, prediction, date):
        """Save a prediction for later evaluation"""
        prediction_file = self.predictions_dir / f"prediction_{date.strftime('%Y%m%d')}.json"
        with open(prediction_file, 'w') as f:
            json.dump(prediction, f, indent=2, default=str)
    
    def evaluate_prediction(self, prediction, actual_result):
        """Evaluate a single prediction"""
        predicted_winner = "HOME" if prediction['predicted_spread'] > 0 else "AWAY"
        actual_winner_ats = "HOME" if actual_result['final_spread'] > prediction['spread'] else "AWAY"
        
        return {
            'date': prediction['date'],
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'predicted_spread': prediction['predicted_spread'],
            'actual_spread': actual_result['final_spread'],
            'betting_spread': prediction['spread'],
            'confidence': prediction['confidence'],
            'correct_ats': predicted_winner == actual_winner_ats,
            'margin_error': abs(prediction['predicted_spread'] - actual_result['final_spread']),
            'key_factors': prediction['key_factors']
        }
    
    def analyze_performance(self, days_back=30):
        """Analyze prediction performance over time"""
        results = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Load all predictions and results in date range
        for day in pd.date_range(start_date, end_date):
            pred_file = self.predictions_dir / f"prediction_{day.strftime('%Y%m%d')}.json"
            if pred_file.exists():
                with open(pred_file, 'r') as f:
                    predictions = json.load(f)
                    for pred in predictions:
                        if 'actual_result' in pred:
                            results.append(self.evaluate_prediction(pred, pred['actual_result']))
        
        if not results:
            return None
            
        df = pd.DataFrame(results)
        
        # Calculate performance metrics
        analysis = {
            'overall': {
                'total_predictions': len(df),
                'accuracy_ats': df['correct_ats'].mean(),
                'average_margin_error': df['margin_error'].mean(),
                'roi': self._calculate_roi(df),
            },
            'by_confidence': self._analyze_by_confidence(df),
            'trends': self._analyze_trends(df),
            'key_factors': self._analyze_key_factors(df)
        }
        
        # Save analysis
        analysis_file = self.performance_dir / f"analysis_{end_date.strftime('%Y%m%d')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _calculate_roi(self, df, unit_bet=100):
        """Calculate ROI assuming unit bet size"""
        wins = df[df['correct_ats']].shape[0]
        losses = df[~df['correct_ats']].shape[0]
        
        # Assuming -110 odds for standard spread bets
        profit = wins * (unit_bet / 1.1) - losses * unit_bet
        investment = len(df) * unit_bet
        
        return profit / investment if investment > 0 else 0
    
    def _analyze_by_confidence(self, df):
        """Analyze performance grouped by confidence levels"""
        confidence_thresholds = CONFIG['model']['confidence_thresholds']
        
        df['confidence_level'] = pd.cut(
            df['confidence'],
            bins=[-float('inf'), confidence_thresholds['low'], confidence_thresholds['medium'], 
                  confidence_thresholds['high'], float('inf')],
            labels=['very_low', 'low', 'medium', 'high']
        )
        
        return df.groupby('confidence_level').agg({
            'correct_ats': ['count', 'mean'],
            'margin_error': 'mean'
        }).to_dict()
    
    def _analyze_trends(self, df):
        """Analyze performance trends over time"""
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date').resample('W').agg({
            'correct_ats': 'mean',
            'margin_error': 'mean'
        })
        
        return {
            'weekly_accuracy': weekly['correct_ats'].tolist(),
            'weekly_margin_error': weekly['margin_error'].tolist(),
            'weeks': weekly.index.strftime('%Y-%m-%d').tolist()
        }
    
    def _analyze_key_factors(self, df):
        """Analyze which key factors are most predictive"""
        factor_performance = {}
        
        for _, row in df.iterrows():
            for factor in row['key_factors']:
                if factor not in factor_performance:
                    factor_performance[factor] = {'correct': 0, 'total': 0}
                factor_performance[factor]['total'] += 1
                if row['correct_ats']:
                    factor_performance[factor]['correct'] += 1
        
        return {
            factor: {
                'accuracy': stats['correct'] / stats['total'],
                'frequency': stats['total'] / len(df)
            }
            for factor, stats in factor_performance.items()
            if stats['total'] >= 10  # Minimum sample size
        }
    
    def get_latest_analysis(self):
        """Get the most recent performance analysis"""
        analysis_files = sorted(self.performance_dir.glob('analysis_*.json'))
        if not analysis_files:
            return None
            
        with open(analysis_files[-1], 'r') as f:
            return json.load(f)
    
    def generate_insights(self):
        """Generate actionable insights from performance data"""
        analysis = self.get_latest_analysis()
        if not analysis:
            return []
            
        insights = []
        
        # Check overall performance
        if analysis['overall']['accuracy_ats'] < 0.5:
            insights.append({
                'type': 'warning',
                'message': 'Overall accuracy below 50%',
                'suggestion': 'Consider model retraining or feature engineering'
            })
        
        # Check confidence levels
        conf_analysis = analysis['by_confidence']
        for level, stats in conf_analysis.items():
            acc = stats['correct_ats']['mean']
            if acc < 0.45:
                insights.append({
                    'type': 'warning',
                    'message': f'Poor performance at {level} confidence',
                    'suggestion': 'Adjust confidence thresholds or feature weights'
                })
            elif acc > 0.6:
                insights.append({
                    'type': 'opportunity',
                    'message': f'Strong performance at {level} confidence',
                    'suggestion': 'Consider increasing bet size for these predictions'
                })
        
        # Analyze key factors
        factor_analysis = analysis['key_factors']
        strong_factors = [f for f, stats in factor_analysis.items() 
                         if stats['accuracy'] > 0.55 and stats['frequency'] > 0.1]
        
        if strong_factors:
            insights.append({
                'type': 'opportunity',
                'message': f'Strong predictive factors identified: {", ".join(strong_factors)}',
                'suggestion': 'Increase weight of these factors in the model'
            })
        
        return insights 