import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import desc
import json

from nothingbutnet.database import init_db, get_session, Game, Team, Prediction

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs/tracking', exist_ok=True)
    
    # Set up logging with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/tracking/performance_tracking_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def save_prediction_to_db(game_id, home_team_id, away_team_id, predicted_spread, 
                         home_win_probability, prediction_date):
    """Save a prediction to the database"""
    init_db()
    session = get_session()
    
    try:
        # Check if prediction already exists
        existing = session.query(Prediction).filter(
            Prediction.game_id == game_id
        ).first()
        
        if existing:
            # Update existing prediction
            existing.predicted_spread = predicted_spread
            existing.home_win_probability = home_win_probability
            existing.prediction_date = prediction_date
        else:
            # Create new prediction
            prediction = Prediction(
                game_id=game_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                predicted_spread=predicted_spread,
                home_win_probability=home_win_probability,
                prediction_date=prediction_date
            )
            session.add(prediction)
        
        session.commit()
        return True
    
    except Exception as e:
        logging.error(f"Error saving prediction: {e}")
        session.rollback()
        return False
    
    finally:
        session.close()

def load_predictions_from_csv():
    """Load predictions from CSV files and save to database"""
    logging.info("Loading predictions from CSV files...")
    
    # Initialize database connection
    init_db()
    session = get_session()
    
    try:
        # Find all prediction CSV files
        predictions_dir = 'data/predictions'
        if not os.path.exists(predictions_dir):
            logging.error(f"Predictions directory not found: {predictions_dir}")
            return
        
        csv_files = [f for f in os.listdir(predictions_dir) if f.startswith('upcoming_games_') and f.endswith('.csv')]
        
        if not csv_files:
            logging.warning("No prediction CSV files found")
            return
        
        logging.info(f"Found {len(csv_files)} prediction files")
        
        # Process each file
        for csv_file in csv_files:
            file_path = os.path.join(predictions_dir, csv_file)
            
            try:
                # Extract date from filename
                date_str = csv_file.replace('upcoming_games_', '').replace('.csv', '')
                prediction_date = datetime.strptime(date_str, '%Y%m%d')
                
                # Load predictions
                predictions_df = pd.read_csv(file_path)
                
                # Process each prediction
                for _, row in predictions_df.iterrows():
                    # Find game in database
                    game_date = datetime.strptime(row['date'], '%Y-%m-%d') if isinstance(row['date'], str) else row['date']
                    
                    # Find home and away teams
                    home_team = session.query(Team).filter(Team.name == row['home_team']).first()
                    away_team = session.query(Team).filter(Team.name == row['away_team']).first()
                    
                    if not home_team or not away_team:
                        logging.warning(f"Team not found: {row['home_team']} or {row['away_team']}")
                        continue
                    
                    # Find game
                    game = session.query(Game).filter(
                        Game.date == game_date,
                        Game.home_team_id == home_team.id,
                        Game.away_team_id == away_team.id
                    ).first()
                    
                    if not game:
                        logging.warning(f"Game not found: {row['home_team']} vs {row['away_team']} on {game_date}")
                        continue
                    
                    # Save prediction
                    save_prediction_to_db(
                        game_id=game.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        predicted_spread=row['predicted_spread'],
                        home_win_probability=row['home_win_probability'],
                        prediction_date=prediction_date
                    )
                
                logging.info(f"Processed predictions from {file_path}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
    
    finally:
        session.close()

def analyze_prediction_accuracy():
    """Analyze prediction accuracy over time"""
    logging.info("Analyzing prediction accuracy...")
    
    # Initialize database connection
    init_db()
    session = get_session()
    
    try:
        # Get all predictions with completed games
        predictions_query = session.query(Prediction, Game).join(
            Game, Prediction.game_id == Game.id
        ).filter(
            Game.home_score.isnot(None),
            Game.away_score.isnot(None)
        ).all()
        
        if not predictions_query:
            logging.warning("No completed games with predictions found")
            return
        
        logging.info(f"Found {len(predictions_query)} completed games with predictions")
        
        # Prepare data for analysis
        results = []
        
        for prediction, game in predictions_query:
            actual_spread = game.home_score - game.away_score
            predicted_spread = prediction.predicted_spread
            
            home_team = session.query(Team).filter(Team.id == game.home_team_id).first()
            away_team = session.query(Team).filter(Team.id == game.away_team_id).first()
            
            result = {
                'date': game.date,
                'home_team': home_team.name if home_team else f"Team_{game.home_team_id}",
                'away_team': away_team.name if away_team else f"Team_{game.away_team_id}",
                'actual_spread': actual_spread,
                'predicted_spread': predicted_spread,
                'error': abs(predicted_spread - actual_spread),
                'correct_winner': (predicted_spread > 0 and actual_spread > 0) or 
                                 (predicted_spread < 0 and actual_spread < 0),
                'home_win_probability': prediction.home_win_probability,
                'actual_home_win': actual_spread > 0,
                'prediction_date': prediction.prediction_date
            }
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        mae = results_df['error'].mean()
        winner_accuracy = results_df['correct_winner'].mean() * 100
        
        logging.info(f"Overall Mean Absolute Error: {mae:.2f} points")
        logging.info(f"Overall Winner Prediction Accuracy: {winner_accuracy:.2f}%")
        
        # Calculate metrics by month
        results_df['month'] = results_df['date'].dt.to_period('M')
        monthly_metrics = results_df.groupby('month').agg({
            'error': 'mean',
            'correct_winner': 'mean'
        })
        monthly_metrics['winner_accuracy'] = monthly_metrics['correct_winner'] * 100
        
        # Save results
        os.makedirs('data/performance', exist_ok=True)
        
        # Save overall metrics
        metrics = {
            'mae': mae,
            'winner_accuracy': winner_accuracy,
            'total_predictions': len(results_df),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('data/performance/overall_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save detailed results
        results_df.to_csv('data/performance/prediction_results.csv', index=False)
        
        # Save monthly metrics
        monthly_metrics.to_csv('data/performance/monthly_metrics.csv')
        
        logging.info("Performance analysis complete")
        
        # Generate visualizations
        create_performance_visualizations(results_df, monthly_metrics)
        
        # Print summary
        print("\nModel Performance Summary:")
        print(f"Total Predictions Analyzed: {len(results_df)}")
        print(f"Mean Absolute Error: {mae:.2f} points")
        print(f"Winner Prediction Accuracy: {winner_accuracy:.2f}%")
        
        print("\nMonthly Performance:")
        print(monthly_metrics[['error', 'winner_accuracy']])
        
    finally:
        session.close()

def create_performance_visualizations(results_df, monthly_metrics):
    """Create visualizations of model performance"""
    logging.info("Creating performance visualizations...")
    
    # Create visualizations directory
    os.makedirs('data/performance/visualizations', exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    try:
        # 1. Error Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['error'], bins=20, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Absolute Error (points)')
        plt.ylabel('Frequency')
        plt.savefig('data/performance/visualizations/error_distribution.png')
        plt.close()
        
        # 2. Actual vs Predicted Spread
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='actual_spread', y='predicted_spread', data=results_df)
        plt.axline([0, 0], [1, 1], color='red', linestyle='--')
        plt.title('Actual vs Predicted Spread')
        plt.xlabel('Actual Spread')
        plt.ylabel('Predicted Spread')
        plt.savefig('data/performance/visualizations/actual_vs_predicted.png')
        plt.close()
        
        # 3. Monthly Performance
        plt.figure(figsize=(12, 6))
        monthly_metrics['winner_accuracy'].plot(kind='bar', color='skyblue')
        plt.title('Monthly Winner Prediction Accuracy')
        plt.xlabel('Month')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/performance/visualizations/monthly_accuracy.png')
        plt.close()
        
        # 4. Monthly MAE
        plt.figure(figsize=(12, 6))
        monthly_metrics['error'].plot(kind='bar', color='salmon')
        plt.title('Monthly Mean Absolute Error')
        plt.xlabel('Month')
        plt.ylabel('MAE (points)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/performance/visualizations/monthly_mae.png')
        plt.close()
        
        # 5. Calibration curve (predicted probability vs actual frequency)
        results_df['prob_bin'] = pd.cut(results_df['home_win_probability'], bins=10)
        calibration = results_df.groupby('prob_bin')['actual_home_win'].mean().reset_index()
        calibration['bin_center'] = calibration['prob_bin'].apply(lambda x: x.mid)
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(calibration['bin_center'], calibration['actual_home_win'], 'o-', label='Model')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Frequency')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/performance/visualizations/calibration_curve.png')
        plt.close()
        
        logging.info("Visualizations created successfully")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")

def track_performance():
    """Main function to track model performance"""
    log_file = setup_logging()
    logging.info("Starting performance tracking")
    logging.info(f"Logs will be saved to: {log_file}")
    
    try:
        # Load predictions from CSV files
        load_predictions_from_csv()
        
        # Analyze prediction accuracy
        analyze_prediction_accuracy()
        
    except Exception as e:
        logging.error(f"Error in performance tracking: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    track_performance() 