import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell
nb.cells.append(nbf.v4.new_markdown_cell("""# NBA Game Data Exploratory Data Analysis (EDA)

This notebook performs exploratory data analysis on the collected NBA game data, including advanced feature engineering and performance metrics."""))

# Add code cells
code_cells = [
    """# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set plot style
plt.style.use('seaborn')
sns.set_palette('viridis')

# Create directories for saving plots and processed data
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../plots', exist_ok=True)""",

    """# Load team data
teams_df = pd.read_csv('../data/teams.csv')
print("Teams Data:")
print(teams_df.info())
teams_df.head()""",

    """# Load game data for both seasons
seasons = ['2022-23', '2023-24']
games_dfs = []

for season in seasons:
    games_file = f'../data/games_{season}.csv'
    if os.path.exists(games_file):
        df = pd.read_csv(games_file)
        df['SEASON'] = season
        games_dfs.append(df)

# Combine all games data
games_df = pd.concat(games_dfs, ignore_index=True)
print("\\nGames Data:")
print(games_df.info())
games_df.head()""",

    """# Basic statistics
print("\\nBasic Statistics:")
print(games_df.describe())

# Check for null values
print("\\nNull Values:")
print(games_df.isnull().sum())""",

    """# Advanced Feature Engineering

# 1. Scoring Efficiency Metrics
games_df['eFG_PCT'] = (games_df['FGM'] + 0.5 * games_df['FG3M']) / games_df['FGA']
games_df['TS_PCT'] = games_df['PTS'] / (2 * (games_df['FGA'] + 0.44 * games_df['FTA']))
games_df['AST_TOV'] = games_df['AST'] / (games_df['TOV'] + 1e-5)  # Adding small value to avoid division by zero

# 2. Contextual Features
games_df['HOME_GAME'] = games_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

# 3. Calculate days since last game
games_df = games_df.sort_values(['TEAM_ID', 'GAME_DATE'])
games_df['DAYS_SINCE_LAST_GAME'] = games_df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days

# 4. Calculate win streaks
games_df['WIN'] = games_df['WL'].apply(lambda x: 1 if x == 'W' else 0)
games_df['WIN_STREAK'] = games_df.groupby('TEAM_ID')['WIN'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
)

# Display new features
print("\\nNew Features:")
print(games_df[['eFG_PCT', 'TS_PCT', 'AST_TOV', 'HOME_GAME', 'DAYS_SINCE_LAST_GAME', 'WIN_STREAK']].head())""",

    """# Distribution of Advanced Metrics
plt.figure(figsize=(15, 10))

# Create subplots for each metric
metrics = ['eFG_PCT', 'TS_PCT', 'AST_TOV']
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.histplot(games_df[metric], kde=True)
    plt.title(f'Distribution of {metric}')
    plt.xlabel(metric)

plt.tight_layout()
plt.savefig('../plots/advanced_metrics_distribution.png')
plt.show()""",

    """# Correlation Analysis of Advanced Features
advanced_features = ['eFG_PCT', 'TS_PCT', 'AST_TOV', 'HOME_GAME', 'DAYS_SINCE_LAST_GAME', 'WIN_STREAK', 'WL']
corr_df = games_df[advanced_features].copy()
corr_df['WL'] = corr_df['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Calculate correlation matrix
corr_matrix = corr_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Advanced Features')
plt.savefig('../plots/advanced_features_correlation.png')
plt.show()""",

    """# Rolling Statistics Analysis
def calculate_rolling_stats(df, window=5):
    rolling_stats = df.sort_values('GAME_DATE').groupby('TEAM_ID').rolling(
        window=window,
        on='GAME_DATE'
    ).agg({
        'PTS': ['mean', 'std'],
        'eFG_PCT': ['mean', 'std'],
        'TS_PCT': ['mean', 'std'],
        'AST_TOV': ['mean', 'std'],
        'REB': ['mean', 'std'],
        'PLUS_MINUS': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    rolling_stats.columns = ['TEAM_ID', 'GAME_DATE'] + [
        f'rolling_{col[0]}_{col[1]}' for col in rolling_stats.columns[2:]
    ]
    return rolling_stats

# Calculate rolling stats for different windows
rolling_5 = calculate_rolling_stats(games_df, window=5)
rolling_10 = calculate_rolling_stats(games_df, window=10)

# Save rolling stats
rolling_5.to_csv('../data/processed/rolling_stats_5.csv', index=False)
rolling_10.to_csv('../data/processed/rolling_stats_10.csv', index=False)""",

    """# Team Performance Analysis
def plot_team_performance(team_id, metrics=['eFG_PCT', 'TS_PCT', 'AST_TOV']):
    team_games = games_df[games_df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    plt.figure(figsize=(15, 5 * len(metrics)))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
        plt.plot(team_games['GAME_DATE'], team_games[metric], label='Game Stats')
        plt.plot(team_games['GAME_DATE'], team_games[metric].rolling(window=5).mean(), 
                label='5-Game Average')
        plt.title(f'{metric} Over Time for Team {team_id}')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'../plots/team_{team_id}_performance.png')
    plt.show()

# Example: Plot performance for a few teams
for team_id in teams_df['id'].head(3):
    plot_team_performance(team_id)""",

    """# Home vs Away Performance Analysis
home_away_stats = games_df.groupby('HOME_GAME').agg({
    'eFG_PCT': 'mean',
    'TS_PCT': 'mean',
    'AST_TOV': 'mean',
    'PTS': 'mean',
    'REB': 'mean',
    'PLUS_MINUS': 'mean'
}).reset_index()

# Plot home vs away performance
plt.figure(figsize=(12, 6))
home_away_stats.set_index('HOME_GAME').plot(kind='bar')
plt.title('Home vs Away Performance Comparison')
plt.xlabel('Home Game (1) vs Away Game (0)')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.legend(title='Metrics')
plt.tight_layout()
plt.savefig('../plots/home_away_comparison.png')
plt.show()""",

    """# Save processed data with new features
games_df.to_csv('../data/processed/processed_games_with_features.csv', index=False)

# Save team statistics
team_stats = games_df.groupby('TEAM_ID').agg({
    'eFG_PCT': ['mean', 'std'],
    'TS_PCT': ['mean', 'std'],
    'AST_TOV': ['mean', 'std'],
    'PTS': ['mean', 'std'],
    'REB': ['mean', 'std'],
    'PLUS_MINUS': ['mean', 'std']
}).reset_index()

team_stats.columns = ['TEAM_ID'] + [f'{col[0]}_{col[1]}' for col in team_stats.columns[1:]]
team_stats.to_csv('../data/processed/team_advanced_stats.csv', index=False)"""
]

for code in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(code))

# Write the notebook to a file
with open('notebooks/02_eda.ipynb', 'w') as f:
    nbf.write(nb, f) 