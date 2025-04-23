import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

class DataProcessor:
    def __init__(self, data_dir='../../data'):
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.processed_data_dir = os.path.join(data_dir, 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Required columns for processing
        self.required_columns = [
            'TEAM_ID', 'GAME_DATE', 'WL', 'PTS', 'FG_PCT', 'FG3_PCT',
            'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS'
        ]
        
        # Initialize imputers
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    def impute_missing_values(self, df):
        """Handle missing values in the dataframe"""
        # Make a copy to avoid modifying the original
        df_imputed = df.copy()
        
        # Handle numeric columns
        numeric_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS']
        df_imputed[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        
        # Handle categorical columns
        categorical_cols = ['WL']
        df_imputed[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        return df_imputed
    
    def validate_data(self, df):
        """Validate that the dataframe has all required columns and data types"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_values = df[self.required_columns].isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            # Impute missing values
            df = self.impute_missing_values(df)
            logging.info("Missing values have been imputed")
        
        return df
    
    def load_raw_data(self, season):
        """Load raw game data for a specific season"""
        try:
            games_file = os.path.join(self.raw_data_dir, f'games_{season}.csv')
            if not os.path.exists(games_file):
                raise FileNotFoundError(f"Games file not found for season {season}")
            
            df = pd.read_csv(games_file)
            df = self.validate_data(df)
            return df
        except Exception as e:
            logging.error(f"Error loading data for season {season}: {str(e)}")
            raise
    
    def calculate_team_stats(self, df):
        """Calculate team-level statistics"""
        try:
            team_stats = df.groupby('TEAM_ID').agg({
                'PTS': ['mean', 'std'],
                'FG_PCT': ['mean', 'std'],
                'FG3_PCT': ['mean', 'std'],
                'FT_PCT': ['mean', 'std'],
                'REB': ['mean', 'std'],
                'AST': ['mean', 'std'],
                'TOV': ['mean', 'std'],
                'PLUS_MINUS': ['mean', 'std']
            }).reset_index()
            
            team_stats.columns = ['TEAM_ID'] + [f'{col[0]}_{col[1]}' for col in team_stats.columns[1:]]
            return team_stats
        except Exception as e:
            logging.error(f"Error calculating team stats: {str(e)}")
            raise
    
    def calculate_rolling_stats(self, df, window=5):
        """Calculate rolling statistics for teams"""
        try:
            # Ensure GAME_DATE is datetime
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            rolling_stats = df.sort_values('GAME_DATE').groupby('TEAM_ID').rolling(
                window=window,
                on='GAME_DATE'
            ).agg({
                'PTS': 'mean',
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean',
                'FT_PCT': 'mean',
                'REB': 'mean',
                'AST': 'mean',
                'TOV': 'mean',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            
            rolling_stats.columns = ['TEAM_ID', 'GAME_DATE'] + [f'rolling_{col}' for col in rolling_stats.columns[2:]]
            return rolling_stats
        except Exception as e:
            logging.error(f"Error calculating rolling stats: {str(e)}")
            raise
    
    def calculate_win_streak(self, df):
        """Calculate win streaks for teams"""
        try:
            df = df.sort_values('GAME_DATE')
            df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
            df['streak'] = df.groupby('TEAM_ID')['WIN'].transform(
                lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
            )
            return df
        except Exception as e:
            logging.error(f"Error calculating win streaks: {str(e)}")
            raise
    
    def process_season_data(self, season):
        """Process data for a specific season"""
        logging.info(f"Processing data for season {season}...")
        
        try:
            # Load raw data
            games_df = self.load_raw_data(season)
            
            # Calculate team stats
            team_stats = self.calculate_team_stats(games_df)
            team_stats.to_csv(os.path.join(self.processed_data_dir, f'team_stats_{season}.csv'), index=False)
            
            # Calculate rolling stats
            rolling_stats = self.calculate_rolling_stats(games_df)
            rolling_stats.to_csv(os.path.join(self.processed_data_dir, f'rolling_stats_{season}.csv'), index=False)
            
            # Calculate win streaks
            games_with_streaks = self.calculate_win_streak(games_df)
            games_with_streaks.to_csv(os.path.join(self.processed_data_dir, f'games_with_streaks_{season}.csv'), index=False)
            
            logging.info(f"Season {season} data processing complete!")
        except Exception as e:
            logging.error(f"Error processing season {season}: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor()
    seasons = ['2022-23', '2023-24']  # Add more seasons as needed
    for season in seasons:
        try:
            processor.process_season_data(season)
        except Exception as e:
            logging.error(f"Failed to process season {season}: {str(e)}")
            continue 