import pandas as pd
import os

def combine_game_files():
    """Combine game data files into a single games.csv file"""
    print("Combining game data files...")
    
    # Read all game files
    game_files = [f for f in os.listdir('data/raw') if f.startswith('games_')]
    dfs = []
    
    for file in game_files:
        df = pd.read_csv(os.path.join('data/raw', file))
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates if any
    combined_df = combined_df.drop_duplicates()
    
    # Sort by date
    combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
    combined_df = combined_df.sort_values('GAME_DATE')
    
    # Save combined file
    output_path = os.path.join('data/raw', 'games.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined game data to {output_path}")
    print(f"Total games: {len(combined_df)}")

if __name__ == "__main__":
    combine_game_files() 