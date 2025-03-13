"""
Baseline linear regression model for ozone concentration prediction.
Based on Model 1 from the research paper using radiation as a single predictor.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import json
import matplotlib.pyplot as plt
import seaborn as sns

class OzoneBaselineModel:
    def __init__(self):
        """Initialize the baseline ozone prediction model."""
        self.model = LinearRegression()
        self.metrics = {}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the linear regression model.
        
        Args:
            X_train (np.ndarray): Training features (radiation)
            y_train (np.ndarray): Training targets (ozone concentration)
        """
        self.model.fit(X_train.reshape(-1, 1), y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features (radiation)
            
        Returns:
            np.ndarray: Predicted ozone concentrations
        """
        return self.model.predict(X.reshape(-1, 1))
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test targets
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': self.model.score(X_test.reshape(-1, 1), y_test),
            'coefficient': float(self.model.coef_[0]),
            'intercept': float(self.model.intercept_)
        }
        
        return self.metrics
    
    def plot_results(self, X_test: np.ndarray, y_test: np.ndarray, 
                    dates_test: pd.DatetimeIndex = None,
                    save_dir: str = "results/figures") -> None:
        """
        Create and save various plots to analyze model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test targets
            dates_test (pd.DatetimeIndex): Test dates for time series plot
            save_dir (str): Directory to save the plots
        """
        y_pred = self.predict(X_test)
        
        # Create figures directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
        plt.plot(X_test, y_pred, color='red', label='Predicted')
        plt.xlabel('Radiation (W/m²)')
        plt.ylabel('Ozone Concentration (µg/m³)')
        plt.title('Baseline Model: Ozone Concentration vs Radiation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "scatter_plot.png"))
        plt.close()
        
        # 2. Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Ozone Concentration (µg/m³)')
        plt.ylabel('Residuals (µg/m³)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "residual_plot.png"))
        plt.close()
        
        # 3. Histogram of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals (µg/m³)')
        plt.ylabel('Count')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "residual_distribution.png"))
        plt.close()
        
        # 4. Time series plot (if dates are provided)
        if dates_test is not None:
            plt.figure(figsize=(15, 6))
            plt.plot(dates_test, y_test, label='Actual', alpha=0.7)
            plt.plot(dates_test, y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Ozone Concentration (µg/m³)')
            plt.title('Ozone Concentration Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "time_series.png"))
            plt.close()
    
    def save_metrics(self, save_path: str) -> None:
        """
        Save model metrics to a JSON file.
        
        Args:
            save_path (str): Path to save the metrics
        """
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

def main():
    """Main function to train and evaluate the baseline model."""
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Load processed data
    processed_dir = "data/processed"
    train_data = pd.read_csv(os.path.join(processed_dir, "train_data.csv"))
    test_data = pd.read_csv(os.path.join(processed_dir, "test_data.csv"))
    
    # Convert date column to datetime
    train_data['date'] = pd.to_datetime(train_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])
    
    # Extract features and targets
    X_train = train_data['radiation'].values
    y_train = train_data['ozone'].values
    X_test = test_data['radiation'].values
    y_test = test_data['ozone'].values
    
    # Initialize and train model
    model = OzoneBaselineModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Evaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f} µg m⁻³")
    print(f"RMSE: {metrics['rmse']:.2f} µg m⁻³")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"\nModel Equation:")
    print(f"[O₃] = {metrics['coefficient']:.3f} · Radiation + {metrics['intercept']:.3f} µg m⁻³")
    
    # Create visualizations
    model.plot_results(X_test, y_test, test_data['date'])
    
    # Save metrics
    model.save_metrics(os.path.join(results_dir, "models", "baseline_metrics.json"))

if __name__ == "__main__":
    main() 