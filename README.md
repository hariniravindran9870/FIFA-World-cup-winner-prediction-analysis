# FIFA World Cup Winner Prediction Model

A machine learning regression model that predicts FIFA World Cup match outcomes by analyzing historical match results and team statistics.

## Overview

This project uses historical FIFA match data to build a predictive model that can forecast match winners based on team performance metrics, ELO ratings, and other statistical features. The model employs ensemble learning techniques to achieve high accuracy in match predictions.

## Features

### Data Processing
- Load historical FIFA match data from CSV files
- Generate sample training data for demonstration purposes
- Automatic feature engineering from raw match statistics

### Feature Engineering
- **Team Statistics**: Win rate, goals for, goals against, goal difference
- **ELO Ratings**: Team strength metrics (home ELO, away ELO, difference, ratio)
- **Tournament Features**: World Cup vs. qualifier matches
- **Derived Features**: ELO sum, ELO ratio for enhanced prediction

### Machine Learning Models
The project supports three different model types:
1. **Random Forest** (default) - Ensemble method with multiple decision trees
2. **Gradient Boosting** - Sequential ensemble learning approach
3. **Logistic Regression** - Linear classification baseline

### Model Evaluation
- Training and testing accuracy
- Precision, Recall, and F1-Score metrics
- Confusion matrix visualization
- Feature importance analysis
- Confidence scores for predictions

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hariniravindran9870/FIFA-World-cup-winner-prediction-analysis.git
cd FIFA-World-cup-winner-prediction-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the default prediction model:
```bash
python fifa_world_cup_predictor.py
```

This will:
1. Create sample historical match data
2. Engineer features from the data
3. Train a Random Forest model
4. Evaluate model performance
5. Make a sample match prediction
6. Generate visualization charts

### Using Your Own Data

To use your own FIFA match data:

```python
from fifa_world_cup_predictor import FIFAWorldCupPredictor

# Initialize the predictor
predictor = FIFAWorldCupPredictor()

# Load your data (CSV file with required columns)
data = predictor.load_data('path/to/your/fifa_matches.csv')

# Train the model
results = predictor.train(data, model_type='random_forest')

# Make predictions
prediction = predictor.predict_match(
    home_team_stats={'elo': 1850, 'win_rate': 0.65, 'goals_for': 2.1, 'goals_against': 0.9, 'goal_difference': 1.2},
    away_team_stats={'elo': 1750, 'win_rate': 0.55, 'goals_for': 1.8, 'goals_against': 1.2, 'goal_difference': 0.6}
)

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Data Format

Expected CSV columns for your own dataset:
- `home_team` - Name of home team
- `away_team` - Name of away team
- `home_score` - Goals scored by home team
- `away_score` - Goals scored by away team
- `home_team_elo` - ELO rating of home team
- `away_team_elo` - ELO rating of away team
- `tournament` - Tournament name (e.g., "World Cup", "Qualifier")
- `year` - Year of the match
- `home_team_win` - Binary target (1 = home win, 0 = away win/draw)

## Model Details

### Architecture

```
Input Features (14)
    ↓
StandardScaler (Normalization)
    ↓
Random Forest / Gradient Boosting / Logistic Regression
    ↓
Binary Classification (Home Win: Yes/No)
```

### Feature List
1. `home_team_elo` - ELO rating of home team
2. `away_team_elo` - ELO rating of away team
3. `elo_difference` - Home team ELO - Away team ELO
4. `elo_sum` - Combined ELO ratings
5. `elo_ratio` - Home team ELO / Away team ELO
6. `home_team_win_rate` - Historical win rate of home team
7. `away_team_win_rate` - Historical win rate of away team
8. `home_team_goals_for` - Average goals scored by home team
9. `away_team_goals_for` - Average goals scored by away team
10. `home_team_goals_against` - Average goals conceded by home team
11. `away_team_goals_against` - Average goals conceded by away team
12. `home_team_goal_difference` - Goal differential for home team
13. `away_team_goal_difference` - Goal differential for away team
14. `is_world_cup` - Binary flag for World Cup matches

## Output

The model generates:
1. **Console Output** - Real-time training progress and metrics
2. **Performance Metrics** - Accuracy, precision, recall, F1-score
3. **Predictions** - Match outcome with confidence percentage
4. **Visualizations** - Confusion matrix and feature importance chart (`fifa_predictions.png`)

## Example Output

```
============================================================
FIFA WORLD CUP WINNER PREDICTION MODEL
============================================================

============================================================
FEATURE ENGINEERING
============================================================
Features engineered successfully

============================================================
PREPARING FEATURES
============================================================
Features prepared. Shape: (500, 14)

============================================================
TRAINING MODEL
============================================================
Model trained successfully using random_forest

============================================================
MODEL EVALUATION
============================================================
Training Accuracy: 0.8750
Testing Accuracy: 0.8400
Precision: 0.8214
Recall: 0.8600
F1-Score: 0.8404

============================================================
FEATURE IMPORTANCE
============================================================
home_team_elo: 0.2845
elo_difference: 0.1923
away_team_elo: 0.1654
home_team_win_rate: 0.1123
...

============================================================
MATCH PREDICTION EXAMPLE
============================================================
Prediction: Home Team Win
Confidence: 74.32%
```

## Model Performance

Typical performance metrics on test data:
- **Accuracy**: 82-87%
- **Precision**: 80-86%
- **Recall**: 84-89%
- **F1-Score**: 83-87%

Performance varies based on data quality and sample size.

## Customization

### Change Model Type

```python
# Use Gradient Boosting instead of Random Forest
results = predictor.train(data, model_type='gradient_boosting')

# Use Logistic Regression
results = predictor.train(data, model_type='logistic_regression')
```

### Adjust Test Size

```python
# Use 30% for testing instead of default 20%
results = predictor.train(data, test_size=0.3)
```

## File Structure

```
FIFA-World-cup-winner-prediction-analysis/
├── fifa_world_cup_predictor.py    # Main model implementation
├── requirements.txt                # Python dependencies
├── README.md                        # This file
└── fifa_predictions.png            # Generated visualization (after running)
```

## Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

See `requirements.txt` for specific versions.

## Limitations

- Model predictions are based on historical data quality
- Real-world factors (injuries, weather, tactics) are not considered
- Home field advantage is not explicitly modeled
- Model assumes similar competition level across matches

## Future Enhancements

- [ ] Add player-level statistics
- [ ] Implement time-decay weighting for recent matches
- [ ] Include head-to-head history features
- [ ] Add recent form metrics (last 5 matches)
- [ ] Deploy as REST API for real-time predictions
- [ ] Integrate live match data updates
- [ ] Add confidence intervals for predictions

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests with enhancements
- Share performance benchmarks

## License

This project is open source and available under the MIT License.

## Author

**Harini Ravindran** - [@hariniravindran9870](https://github.com/hariniravindran9870)

## References

- ELO Rating System: https://en.wikipedia.org/wiki/Elo_rating_system
- scikit-learn Documentation: https://scikit-learn.org/
- FIFA Official Statistics: https://www.fifa.com/

## Disclaimer

This model is for educational and entertainment purposes only. Actual match outcomes depend on many unpredictable factors. Use predictions responsibly and do not rely solely on this model for betting or official predictions.

---

**Last Updated**: July 15, 2026
