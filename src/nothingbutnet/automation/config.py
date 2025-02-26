import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
HYPOTHESIS_DIR = BASE_DIR / "hypotheses"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "automation.log"

# Task scheduling (in crontab format)
SCHEDULE = {
    "code_review": "0 */12 * * *",      # Twice daily
    "data_quality": "30 8 * * *",        # Daily at 8:30 AM
    "hypothesis_generation": "0 9 * * 1", # Weekly on Monday at 9 AM
    "model_evaluation": "0 10 * * *",     # Daily at 10 AM
    "performance_tracking": "0 */4 * * *", # Every 4 hours
    "feature_suggestion": "0 11 * * 1",    # Weekly on Monday at 11 AM
    "documentation": "0 0 1 * *",         # Monthly
}

# Claude API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Project paths to monitor
PROJECT_PATHS = {
    "data_collector": "../src/NBADataCollector.py",
    "data_dir": "../data",
    "models_dir": "../models",
    "predictions_dir": "../predictions",
    "hypothesis_dir": "../hypotheses"
}

# Task types and their descriptions
TASK_TYPES = {
    "code_review": "Review and suggest improvements to the codebase",
    "data_quality": "Analyze and improve data collection and processing",
    "hypothesis_generation": "Generate and prioritize new prediction hypotheses",
    "model_evaluation": "Evaluate model performance and suggest improvements",
    "performance_tracking": "Track and analyze prediction performance",
    "feature_suggestion": "Suggest new features and enhancements",
    "documentation": "Improve and update documentation"
}

# Model configuration
MODEL_CONFIG = {
    "confidence_thresholds": {
        "high": 0.7,
        "medium": 0.6,
        "low": 0.5
    },
    "bet_sizing": {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.5
    },
    "performance_metrics": [
        "accuracy",
        "roi",
        "kelly_criterion",
        "sharp_ratio"
    ]
}

# Data collection configuration
DATA_CONFIG = {
    "update_frequency": "daily",
    "historical_seasons": range(2015, 2024),
    "required_features": [
        "team_stats",
        "player_stats",
        "betting_lines",
        "injuries",
        "rest_days"
    ]
} 