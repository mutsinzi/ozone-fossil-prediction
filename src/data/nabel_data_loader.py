"""
Module for loading and preprocessing NABEL air quality data from Switzerland.
Specifically handles data from the Lugano-Università station for ozone prediction.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
from sklearn.model_selection import train_test_split

class NABELDataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the NABEL data loader.
        
        Args:
            data_dir (str): Directory where raw NABEL data files are stored
        """
        self.data_dir = data_dir
        self.station_name = "Lugano-Università"
        
    def load_and_prepare_data(self, start_year: int = 2016, 
                            end_year: int = 2023) -> pd.DataFrame:
        """
        Load and prepare NABEL data for the specified time period.
        
        Args:
            start_year (int): Start year for data loading
            end_year (int): End year for data loading
            
        Returns:
            pd.DataFrame: Prepared dataset with daily averages
        """
        # Load O3 data
        o3_file = os.path.join(self.data_dir, "O3.csv")
        o3_df = pd.read_csv(o3_file, sep=';', skiprows=4, encoding='latin1')
        o3_df.columns = ['date', 'ozone']
        
        # Load radiation data
        rad_file = os.path.join(self.data_dir, "RAD.csv")
        rad_df = pd.read_csv(rad_file, sep=';', skiprows=4, encoding='latin1')
        rad_df.columns = ['date', 'radiation']
        
        # Convert dates to datetime
        o3_df['date'] = pd.to_datetime(o3_df['date'], format='%d.%m.%Y')
        rad_df['date'] = pd.to_datetime(rad_df['date'], format='%d.%m.%Y')
        
        # Merge O3 and radiation data
        df = pd.merge(o3_df, rad_df, on='date', how='inner')
        
        # Filter by date range
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        df = df[mask].copy()
        
        # Convert values to numeric, handling any non-numeric values
        df['ozone'] = pd.to_numeric(df['ozone'], errors='coerce')
        df['radiation'] = pd.to_numeric(df['radiation'], errors='coerce')
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    
    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw NABEL data to create daily averages.
        
        Args:
            df (pd.DataFrame): Raw data from NABEL
            
        Returns:
            pd.DataFrame: Processed data with daily averages
        """
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Handle missing values
        df = df.dropna()  # For baseline, we simply drop missing values
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets, maintaining temporal order.
        
        Args:
            df (pd.DataFrame): Processed daily data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets
        """
        # Calculate split point to maintain temporal order
        split_idx = int(len(df) * (1 - test_size))
        
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        return train_data, test_data

def main():
    """
    Main function to demonstrate usage of the NABELDataLoader class.
    """
    loader = NABELDataLoader()
    
    try:
        # Load and prepare data
        print("Loading and preparing data...")
        data = loader.load_and_prepare_data()
        
        # Process data
        print("Processing data...")
        daily_data = loader.process_raw_data(data)
        
        # Split data
        print("Splitting data into train and test sets...")
        train_data, test_data = loader.split_data(daily_data)
        
        # Save processed data
        print("Saving processed data...")
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_dir, "train_data.csv"))
        test_data.to_csv(os.path.join(processed_dir, "test_data.csv"))
        
        print("\nData processing completed successfully!")
        print(f"Total samples: {len(daily_data)}")
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Print data summary
        print("\nData Summary:")
        print(daily_data.describe())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 