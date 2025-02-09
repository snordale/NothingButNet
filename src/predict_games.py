import pandas as pd
from NBASpreadPredictor import NBASpreadPredictor
from NBADataCollector import NBADataCollector
import logging

def predict_upcoming_games():
    # Load the trained model
    predictor = NBASpreadPredictor()
    predictor.load_model()
    
    # Get latest data
    collector = NBADataCollector()
    games_df, advanced_df = collector.fetch_basketball_reference()
    players_df = collector.fetch_player_stats(2024)
    standings_df = collector.fetch_standings()
    
    # Get upcoming games (you'll need to implement this)
    upcoming_games = collector.fetch_upcoming_games()
    
    predictions = []
    for _, game in upcoming_games.iterrows():
        try:
            # Prepare features for this game
            game_features = predictor.prepare_game_features(
                games_df, 
                players_df, 
                standings_df,
                game['home_team'],
                game['away_team'],
                game['date']
            )
            
            # Make prediction
            predicted_spread = predictor.predict_spread(game_features)
            
            predictions.append({
                'date': game['date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_spread': predicted_spread
            })
            
        except Exception as e:
            logging.error(f"Error predicting game {game}: {e}")
            continue
    
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    predictions_df = predict_upcoming_games()
    print("\nPredicted Spreads:")
    print(predictions_df)
    
    # Save predictions
    predictions_df.to_csv('predictions.csv', index=False) 