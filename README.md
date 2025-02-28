# NothingButNet
Deep learning project to predict NBA game outcomes against the spread.

## Overview
NothingButNet is a machine learning system that predicts NBA game outcomes against the spread. The system uses historical data, recent performance, and various advanced statistics to generate predictions with confidence levels and betting recommendations.

## Features
- Deep learning model for spread prediction with feature importance tracking
- Advanced statistics including offensive/defensive ratings, rebounding, and turnover metrics
- Multi-source data collection (Basketball Reference, ESPN, Kaggle)
- Performance tracking and analysis
- Confidence-based betting recommendations
- SQLite database for reliable data storage
- Local data caching for efficiency

## Installation

1. Clone the repository:
```bash
git clone https://github.com/snordale/NothingButNet.git
cd NothingButNet
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# See .env.example for required variables
```

## Usage

### Load Initial Data
```bash
# Load data from Kaggle dataset
python3 scripts/load_kaggle_data.py
```

### Train the Model
```bash
# Train the prediction model
python3 scripts/train_model.py
```

### Get Predictions
```bash
# Get predictions for today
python3 -m nothingbutnet.predict

# Get predictions for a specific date
python3 -m nothingbutnet.predict --date YYYY-MM-DD
```

## Project Structure
```
NothingButNet/
├── src/
│   └── nothingbutnet/
│       ├── models/           # Neural network models
│       ├── predictors/       # Prediction strategies
│       ├── automation/       # Automation scripts
│       ├── data_collector.py # Data collection
│       ├── predict.py        # Prediction interface
│       ├── database.py       # Database operations
│       └── config.py         # Configuration
├── data/
│   ├── models/             # Trained models
│   ├── cache/              # Cached responses
│   └── predictions/        # Prediction history
├── logs/                   # Logs and reports
├── scripts/               # Setup and utility scripts
├── .env                   # Environment variables (git-ignored)
├── .env.example          # Environment template
├── requirements.txt       # Dependencies
└── README.md
```

## Model Features
The prediction model uses a comprehensive set of features:

1. **Basic Team Stats**:
   - Win percentage
   - Points per game
   - Points allowed per game
   - Last 10 games performance

2. **Advanced Stats**:
   - Offensive rating (points per 100 possessions)
   - Defensive rating (points allowed per 100 possessions)
   - Net rating
   - Pace

3. **Rebounding Stats**:
   - Offensive rebounds per game
   - Defensive rebounds per game
   - Total rebounds per game
   - Offensive/defensive rebound percentages

4. **Game Context**:
   - Home court advantage
   - Rest days
   - Turnover rates

## Model Architecture
The current model uses a deep neural network with:
- Batch normalization for better training stability
- Dropout layers for regularization
- Feature importance tracking
- L1/L2 regularization
- Learning rate scheduling
- Early stopping

## Data Collection Strategy
The system employs a robust data collection approach:

1. **Multiple Data Sources**:
   - Basketball Reference (primary source)
   - ESPN (backup source)
   - Kaggle datasets (supplementary)

2. **Rate Limiting**:
   - Basketball Reference: 6 seconds between requests
   - ESPN: 3 seconds between requests
   - Exponential backoff for failed requests

3. **Data Caching**:
   - Local caching of responses
   - 24-hour cache validity
   - Reduces unnecessary API calls

## Performance Tracking
The system tracks:
- Prediction accuracy against the spread
- Feature importance rankings
- Model confidence levels
- Key factors influencing predictions

## Security Notice
This project uses environment variables for sensitive configuration. Never commit:
- API keys
- Database credentials
- Private tokens
- Personal configuration

## Data Sources
- Basketball Reference (primary)
  - Game schedules and results
  - Team advanced statistics
  - Historical data
- ESPN (backup)
  - Live game data
  - Team statistics
- Kaggle datasets (supplementary)
  - Historical game data
  - Team and player statistics

**Important**: Never commit sensitive information such as API keys or credentials. Always use environment variables for sensitive data.