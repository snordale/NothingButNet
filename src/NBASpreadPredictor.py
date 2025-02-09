import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime, timedelta

class NBASpreadPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def calculate_rest_days(self, games_df, team_id, game_date):
        """Calculate days of rest since last game"""
        previous_game = games_df[
            (games_df['team_id'] == team_id) & 
            (games_df['date'] < game_date)
        ].sort_values('date').iloc[-1]
        
        return (game_date - previous_game['date']).days

    def get_head_to_head_stats(self, games_df, home_team, away_team, game_date, n_games=3):
        """Get stats from previous matchups"""
        h2h_games = games_df[
            (
                ((games_df['home_team_id'] == home_team) & (games_df['away_team_id'] == away_team)) |
                ((games_df['home_team_id'] == away_team) & (games_df['away_team_id'] == home_team))
            ) &
            (games_df['date'] < game_date)
        ].sort_values('date', ascending=False).head(n_games)
        
        return {
            'avg_point_diff': h2h_games['point_differential'].mean(),
            'home_team_wins': sum(
                (h2h_games['home_team_id'] == home_team) & (h2h_games['home_team_won']) |
                (h2h_games['away_team_id'] == home_team) & (~h2h_games['home_team_won'])
            )
        }

    def prepare_team_stats(self, games_df, team_id, game_date):
        """Calculate comprehensive team statistics"""
        team_games = games_df[
            (games_df['team_id'] == team_id) & 
            (games_df['date'] < game_date)
        ].sort_values('date')
        
        # Season stats
        season_stats = team_games.agg({
            'points': 'mean',
            'points_allowed': 'mean',
            'fg_percentage': 'mean',
            'three_pt_percentage': 'mean',
            'ft_percentage': 'mean',
            'offensive_rebounds': 'mean',
            'defensive_rebounds': 'mean',
            'assists': 'mean',
            'turnovers': 'mean',
            'steals': 'mean',
            'blocks': 'mean',
            'won': 'mean',  # win percentage
            'point_differential': 'mean'
        })
        
        # Last 10 games stats
        last_10_stats = team_games.tail(10).agg({
            'points': 'mean',
            'points_allowed': 'mean',
            'fg_percentage': 'mean',
            'won': 'mean',
            'point_differential': 'mean'
        })
        
        # Advanced stats
        possessions = team_games['fg_attempts'] - team_games['offensive_rebounds'] + \
                     team_games['turnovers'] + (0.4 * team_games['ft_attempts'])
        
        offensive_rating = team_games['points'] / possessions * 100
        defensive_rating = team_games['points_allowed'] / possessions * 100
        
        advanced_stats = {
            'net_rating': offensive_rating.mean() - defensive_rating.mean(),
            'pace': possessions.mean(),
            'true_shooting_pct': team_games['points'] / (2 * (team_games['fg_attempts'] + 0.44 * team_games['ft_attempts'])),
            'effective_fg_pct': (team_games['fg_made'] + 0.5 * team_games['three_pt_made']) / team_games['fg_attempts'],
            'rebound_rate': (team_games['offensive_rebounds'] + team_games['defensive_rebounds']) / \
                          (team_games['offensive_rebounds'] + team_games['defensive_rebounds'] + \
                           team_games['opp_offensive_rebounds'] + team_games['opp_defensive_rebounds']),
            'assist_rate': team_games['assists'] / team_games['fg_made']
        }
        
        return pd.concat([season_stats, last_10_stats, pd.Series(advanced_stats)])

    def prepare_player_stats(self, players_df, team_id, game_date):
        """Calculate player-level statistics"""
        team_players = players_df[
            (players_df['team_id'] == team_id) & 
            (players_df['date'] < game_date)
        ]
        
        # Top 3 scorers
        top_scorers = team_players.sort_values('points', ascending=False).head(3)
        top_scorers_stats = top_scorers.agg({
            'points': 'mean',
            'minutes': 'mean',
            'plus_minus': 'mean',
            'player_efficiency': 'mean'
        })
        
        # Starter availability (assuming starters are top 5 in minutes)
        starters = team_players.sort_values('minutes', ascending=False).head(5)
        starter_availability = starters['is_available'].mean()
        
        return pd.concat([top_scorers_stats, pd.Series({'starter_availability': starter_availability})])

    def prepare_game_features(self, games_df, players_df, standings_df):
        """Prepare features for each game"""
        features = []
        spreads = []
        
        for _, game in games_df.iterrows():
            home_team = game['home_team_id']
            away_team = game['away_team_id']
            game_date = game['date']
            
            try:
                # Team stats
                home_stats = self.prepare_team_stats(games_df, home_team, game_date)
                away_stats = self.prepare_team_stats(games_df, away_team, game_date)
                
                # Player stats
                home_player_stats = self.prepare_player_stats(players_df, home_team, game_date)
                away_player_stats = self.prepare_player_stats(players_df, away_team, game_date)
                
                # Rest days
                home_rest = self.calculate_rest_days(games_df, home_team, game_date)
                away_rest = self.calculate_rest_days(games_df, away_team, game_date)
                
                # Head-to-head stats
                h2h_stats = self.get_head_to_head_stats(games_df, home_team, away_team, game_date)
                
                # Standings info
                home_standing = standings_df[
                    (standings_df['team_id'] == home_team) & 
                    (standings_df['date'] < game_date)
                ].iloc[-1]
                
                away_standing = standings_df[
                    (standings_df['team_id'] == away_team) & 
                    (standings_df['date'] < game_date)
                ].iloc[-1]
                
                # Combine all features
                game_features = {
                    **{f'home_{k}': v for k, v in home_stats.items()},
                    **{f'away_{k}': v for k, v in away_stats.items()},
                    **{f'home_player_{k}': v for k, v in home_player_stats.items()},
                    **{f'away_player_{k}': v for k, v in away_player_stats.items()},
                    'home_rest_days': home_rest,
                    'away_rest_days': away_rest,
                    'h2h_point_diff': h2h_stats['avg_point_diff'],
                    'h2h_home_wins': h2h_stats['home_team_wins'],
                    'home_standing': home_standing['position'],
                    'away_standing': away_standing['position'],
                    'home_games_behind': home_standing['games_behind'],
                    'away_games_behind': away_standing['games_behind'],
                    'is_back_to_back_home': home_rest == 0,
                    'is_back_to_back_away': away_rest == 0
                }
                
                features.append(game_features)
                spreads.append(game['home_team_points'] - game['away_team_points'])
                
            except (IndexError, KeyError):
                # Skip games without sufficient historical data
                continue
            
        return pd.DataFrame(features), np.array(spreads)

    def build_model(self, input_dim):
        """Build neural network for spread prediction"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)  # Linear activation for spread prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error
        )
        
        return model

    def train(self, games_df, players_df, standings_df):
        """Train the model"""
        # Prepare features
        X, y = self.prepare_game_features(games_df, players_df, standings_df)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = self.build_model(X_train.shape[1])
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_mae = self.model.evaluate(X_test_scaled, y_test)
        print(f"\nTest MAE: {test_mae:.1f} points")
        
        return history, test_mae

    def predict_spread(self, game_features):
        """Predict spread for a single game"""
        features_scaled = self.scaler.transform(game_features)
        predicted_spread = self.model.predict(features_scaled)[0][0]
        return predicted_spread