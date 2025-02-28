from pathlib import Path
import yaml
import os
from datetime import datetime

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PERFORMANCE_DIR = DATA_DIR / "performance"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, PERFORMANCE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "data_collection": {
        "seasons": list(range(datetime.now().year - 9, datetime.now().year + 1)),
        "request_delay": 3,
        "cache_duration_hours": 24,
        "data_sources": ["basketball_reference"],
        "required_data": {
            "games": True,
            "players": False,
            "standings": True,
            "betting_lines": True
        }
    },
    "model": {
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10,
        "learning_rate": 0.001,
        "train_test_split": 0.2,
        "validation_split": 0.2,
        "min_games_for_prediction": 5,
        "confidence_thresholds": {
            "high": 0.7,
            "medium": 0.6,
            "low": 0.5
        }
    },
    "features": {
        "lookback_games": 10,
        "h2h_games": 3,
        "use_advanced_stats": True,
        "use_player_stats": True,
        "use_injuries": True
    },
    "prediction": {
        "min_confidence_for_bet": 0.6,
        "max_daily_bets": 3,
        "track_metrics": [
            "accuracy",
            "roi",
            "kelly_criterion",
            "sharp_ratio"
        ]
    },
    "paths": {
        "raw_data": str(DATA_DIR / "raw"),
        "processed_data": str(DATA_DIR / "processed"),
        "models": str(MODELS_DIR),
        "logs": str(LOGS_DIR),
        "predictions": str(DATA_DIR / "predictions"),
        "performance": str(PERFORMANCE_DIR)
    },
    "automation": {
        "update_frequency": "daily",
        "update_time": "10:00",  # Before markets open
        "retrain_frequency": "weekly",
        "performance_analysis_frequency": "daily"
    }
}

def load_config():
    """Load configuration from file or create default"""
    config_path = BASE_DIR / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults to ensure all keys exist
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            return merged
    else:
        # Save default config
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
        return DEFAULT_CONFIG

# Global configuration instance
CONFIG = load_config() 