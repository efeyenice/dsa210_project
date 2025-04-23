import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from data_collection.nba_data_collector import NBADataCollector
from utils.data_processor import DataProcessor

def main():
    print("Starting NBA Game Outcome Prediction Pipeline...")
    
    # Initialize data collector and processor
    collector = NBADataCollector()
    processor = DataProcessor()
    
    # Collect data
    print("\n=== Data Collection Phase ===")
    collector.collect_all_data()
    
    # Process data
    print("\n=== Data Processing Phase ===")
    seasons = ['2022-23', '2023-24']  # Add more seasons as needed
    for season in seasons:
        processor.process_season_data(season)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 