# NothingButNet
Deep learning project to predict basketball game outcomes.

## Overview
NothingButNet is a machine learning system that predicts NBA game outcomes against the spread. The system uses historical data, recent performance, and various other factors to generate predictions with confidence levels and betting recommendations.

## Features
- Deep learning model for spread prediction
- Comprehensive data collection from multiple sources
- Performance tracking and analysis
- Automated daily updates and model retraining
- Confidence-based betting recommendations
- Detailed insights and reporting
- PostgreSQL database for reliable data storage
- Multi-source data collection with rate limiting
- Local data caching for efficiency

## Security Notice
This project uses environment variables for sensitive configuration. Never commit:
- API keys
- Database credentials
- Private tokens
- Personal configuration

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

4. Install PostgreSQL:
```bash
# macOS (using Homebrew)
brew install postgresql


# Start PostgreSQL service
# macOS
brew services start postgresql

```

5. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# See .env.example for required variables
```

## Usage

### Database Setup
```bash
# Initialize the database
python3 scripts/setup_db.py
```

### Get Today's Predictions
```bash
python3 -m nothingbutnet.predict
```

### Get Predictions for a Specific Date
```bash
python3 -m nothingbutnet.predict --date YYYY-MM-DD
```

### Set Up Automated Updates
```bash
python3 -m nothingbutnet.automation.daily_update --setup
```

### Manual Model Retraining
```bash
python3 -m nothingbutnet.automation.daily_update --retrain
```

## Configuration

### Environment Variables
Required and optional environment variables are documented in `.env.example`. Copy this file to `.env` and configure your values:
- Required: Database configuration
- Optional: API keys for additional features

### System Configuration
The system can be configured by editing `config/config.yaml`. Key settings include:
- Data collection parameters
- Model hyperparameters
- Confidence thresholds
- Automation schedules
- Performance tracking metrics

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
│   ├── raw/                 # Raw collected data
│   ├── processed/           # Processed datasets
│   ├── models/             # Trained models
│   ├── cache/              # Cached responses
│   └── predictions/        # Prediction history
├── logs/                   # Logs and reports
├── config/                 # Configuration files
├── scripts/               # Setup and utility scripts
├── .env                   # Environment variables (git-ignored)
├── .env.example          # Environment template
├── requirements.txt       # Dependencies
└── README.md
```

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
   - Random jitter to avoid synchronized requests

3. **Data Caching**:
   - Local caching of responses
   - 24-hour cache validity
   - Reduces unnecessary API calls
   - Improves response time

4. **Error Handling**:
   - Automatic failover between sources
   - Comprehensive error logging
   - Retry mechanism with backoff

## Performance Tracking
The system maintains detailed performance records and generates daily reports including:
- Overall accuracy against the spread
- ROI analysis
- Performance by confidence level
- Key insights and trends
- Automated alerts for significant patterns

Reports can be found in `logs/reports/`.

## Betting Strategy
The system uses a conservative approach to betting recommendations:
1. Only bets with confidence above configurable threshold
2. Uses Kelly criterion for bet sizing
3. Considers historical performance in similar situations
4. Tracks and adapts to performance patterns

## Data Sources
- Basketball Reference (primary)
  - Game schedules and results
  - Team advanced statistics
  - Player statistics
  - Historical data
- ESPN (backup)
  - Live game data
  - Team statistics
  - Alternative data source
- NBA.com (backup)
  - Live game data
  - Team statistics
- Kaggle datasets (supplementary)
- Live betting lines
- Player statistics and injury reports

**Important**: Never commit sensitive information such as API keys or credentials. Always use environment variables for sensitive data.

## License

## Basketball Reference Notes
- Maximum 20 requests per minute
- 6-second delay between requests recommended
- Rate limits apply regardless of bot type
- Use caching to minimize requests

# Goal
Predict the outcomes of a night's NBA and NCAA games against the spread.

If you are unsure what to do next, do whatever is necessary to improve the current model, develop a new model, or make making predictions easier.

# Outcome
I want to be able to run a script with a date as arguement, and have the picks and spreads for each game printed to stdout.

A core part of this project will be automated and it will be in charge of improving itself by generating hypotheses, testing them, saving the results, and then using the results to generate new hypotheses.

Any strategy is on the table, but when I personally pick games I most value teams' recent performances (last 10 games) ATS.

I think deep learning would be the most interesting, but there could be a data constraint.

I leave the direction up to the AI, which should keep records of its performance and notes so that it can optimize its strategy.

# Package Manager
I use venv and pip3 to manage dependencies.

# Database
I use PostgreSQL.
Don't delete any of the data in the database.

# Data Sources
I use Basketball Reference, ESPN, NBA.com, and Kaggle.