"""
FIFA World Cup Winner Prediction Model
Predicts World Cup winners by analyzing historical match results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class FIFAWorldCupPredictor:
    """
    A machine learning model to predict FIFA World Cup winners
    based on historical match data and team statistics.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_data = None
        self.test_data = None
        
    def load_data(self, filepath):
        """
        Load historical FIFA match data
        Expected columns: home_team, away_team, home_score, away_score, 
                         tournament, year, home_team_elo, away_team_elo
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
    
    def create_sample_data(self):
        """
        Create sample FIFA World Cup historical data for demonstration
        """
        np.random.seed(42)
        
        teams = ['Brazil', 'Germany', 'France', 'Italy', 'Argentina', 
                 'Spain', 'Netherlands', 'England', 'Belgium', 'Portugal']
        
        data = []
        for _ in range(500):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            home_score = np.random.randint(0, 5)
            away_score = np.random.randint(0, 5)
            
            # Determine winner (1 = home win, 0 = away win or draw)
            winner = 1 if home_score > away_score else 0
            
            # Simulate team strength (ELO ratings)
            home_elo = np.random.uniform(1400, 2000)
            away_elo = np.random.uniform(1400, 2000)
            
            # More skilled teams tend to score more
            if home_elo > away_elo:
                home_score = max(home_score, home_score + np.random.randint(0, 2))
            
            data.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'home_team_elo': home_elo,
                'away_team_elo': away_elo,
                'tournament': 'World Cup' if np.random.random() > 0.5 else 'Qualifier',
                'year': np.random.randint(2000, 2023),
                'home_team_win': winner
            })
        
        df = pd.DataFrame(data)
        print(f"Sample data created. Shape: {df.shape}")
        return df
    
    def engineer_features(self, data):
        """
        Create feature engineering for the model
        """
        df = data.copy()
        
        # Calculate team statistics from historical data
        team_stats = {}
        
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            home_games = df[df['home_team'] == team]
            away_games = df[df['away_team'] == team]
            
            home_wins = len(home_games[home_games['home_team_win'] == 1])
            away_wins = len(away_games[away_games['home_team_win'] == 0])
            
            home_goals_for = home_games['home_score'].sum()
            home_goals_against = home_games['away_score'].sum()
            away_goals_for = away_games['away_score'].sum()
            away_goals_against = away_games['home_score'].sum()
            
            total_games = len(home_games) + len(away_games)
            win_rate = (home_wins + away_wins) / total_games if total_games > 0 else 0
            
            team_stats[team] = {
                'win_rate': win_rate,
                'goals_for': (home_goals_for + away_goals_for) / total_games if total_games > 0 else 0,
                'goals_against': (home_goals_against + away_goals_against) / total_games if total_games > 0 else 0,
                'goal_difference': ((home_goals_for + away_goals_for) - (home_goals_against + away_goals_against)) / total_games if total_games > 0 else 0
            }
        
        # Add team statistics to dataframe
        for stat in ['win_rate', 'goals_for', 'goals_against', 'goal_difference']:
            df[f'home_team_{stat}'] = df['home_team'].map(lambda x: team_stats.get(x, {}).get(stat, 0))
            df[f'away_team_{stat}'] = df['away_team'].map(lambda x: team_stats.get(x, {}).get(stat, 0))
        
        # ELO-based features
        df['elo_difference'] = df['home_team_elo'] - df['away_team_elo']
        df['elo_sum'] = df['home_team_elo'] + df['away_team_elo']
        df['elo_ratio'] = df['home_team_elo'] / (df['away_team_elo'] + 1)  # Avoid division by zero
        
        # Tournament-based feature
        df['is_world_cup'] = (df['tournament'] == 'World Cup').astype(int)
        
        return df, team_stats
    
    def prepare_features(self, data):
        """
        Select features for the model
        """
        feature_columns = [
            'home_team_elo', 'away_team_elo', 'elo_difference', 'elo_sum', 'elo_ratio',
            'home_team_win_rate', 'away_team_win_rate',
            'home_team_goals_for', 'away_team_goals_for',
            'home_team_goals_against', 'away_team_goals_against',
            'home_team_goal_difference', 'away_team_goal_difference',
            'is_world_cup'
        ]
        
        X = data[feature_columns].fillna(0)
        y = data['home_team_win']
        
        self.feature_names = feature_columns
        return X, y
    
    def train(self, data, test_size=0.2, model_type='random_forest'):
        """
        Train the prediction model
        
        Args:
            data: DataFrame with historical match data
            test_size: Proportion of data to use for testing
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
        """
        # Feature engineering
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        data, team_stats = self.engineer_features(data)
        print(f"Features engineered successfully")
        
        # Prepare features
        print("\n" + "="*60)
        print("PREPARING FEATURES")
        print("="*60)
        X, y = self.prepare_features(data)
        print(f"Features prepared. Shape: {X.shape}")
        print(f"Features: {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model.fit(X_train_scaled, y_train)
        print(f"Model trained successfully using {model_type}")
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE")
            print("="*60)
            importances = self.model.feature_importances_
            for name, importance in sorted(zip(self.feature_names, importances), 
                                          key=lambda x: x[1], reverse=True)[:10]:
                print(f"{name}: {importance:.4f}")
        
        self.training_data = (X_train_scaled, y_train)
        self.test_data = (X_test_scaled, y_test)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict_match(self, home_team_stats, away_team_stats):
        """
        Predict the outcome of a specific match
        
        Args:
            home_team_stats: Dictionary with team statistics
            away_team_stats: Dictionary with team statistics
        
        Returns:
            Prediction probability for home team win
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Create feature vector
        features = np.array([
            home_team_stats.get('elo', 1600),
            away_team_stats.get('elo', 1600),
            home_team_stats.get('elo', 1600) - away_team_stats.get('elo', 1600),
            home_team_stats.get('elo', 1600) + away_team_stats.get('elo', 1600),
            home_team_stats.get('elo', 1600) / (away_team_stats.get('elo', 1600) + 1),
            home_team_stats.get('win_rate', 0.5),
            away_team_stats.get('win_rate', 0.5),
            home_team_stats.get('goals_for', 1.5),
            away_team_stats.get('goals_for', 1.5),
            home_team_stats.get('goals_against', 1.5),
            away_team_stats.get('goals_against', 1.5),
            home_team_stats.get('goal_difference', 0),
            away_team_stats.get('goal_difference', 0),
            1  # is_world_cup
        ]).reshape(1, -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0][1]
        prediction = self.model.predict(features_scaled)[0]
        
        return {
            'prediction': 'Home Team Win' if prediction == 1 else 'Away Team Win/Draw',
            'confidence': max(probability, 1 - probability)
        }
    
    def visualize_results(self):
        """
        Visualize model performance
        """
        if self.test_data is None:
            print("No test data available for visualization")
            return
        
        X_test, y_test = self.test_data
        y_pred = self.model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            axes[1].barh(top_features, top_importances)
            axes[1].set_title('Top 10 Feature Importance')
            axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('fifa_predictions.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'fifa_predictions.png'")
        plt.show()


def main():
    """
    Main function to demonstrate the FIFA World Cup predictor
    """
    print("\n" + "="*60)
    print("FIFA WORLD CUP WINNER PREDICTION MODEL")
    print("="*60)
    
    # Initialize predictor
    predictor = FIFAWorldCupPredictor()
    
    # Load or create data
    # You can replace this with your own data file:
    # data = predictor.load_data('fifa_matches.csv')
    data = predictor.create_sample_data()
    
    # Train model
    results = predictor.train(data, model_type='random_forest')
    
    # Example: Predict a match
    print("\n" + "="*60)
    print("MATCH PREDICTION EXAMPLE")
    print("="*60)
    
    home_team = {
        'elo': 1850,
        'win_rate': 0.65,
        'goals_for': 2.1,
        'goals_against': 0.9,
        'goal_difference': 1.2
    }
    
    away_team = {
        'elo': 1750,
        'win_rate': 0.55,
        'goals_for': 1.8,
        'goals_against': 1.2,
        'goal_difference': 0.6
    }
    
    prediction = predictor.predict_match(home_team, away_team)
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    # Visualize
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    predictor.visualize_results()
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
