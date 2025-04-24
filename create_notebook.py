import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell
nb.cells.append(nbf.v4.new_markdown_cell("""# NBA Data Collection

This notebook collects NBA game data from the NBA API and saves it to CSV files."""))

# Add code cells
code_cells = [
    """# Import required libraries
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, teamgamelog
from nba_api.stats.static import teams, players
import time
from tqdm import tqdm
import os""",

    """# Create data directory if it doesn't exist
data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)""",

    """# Get all NBA teams
teams_df = pd.DataFrame(teams.get_teams())
teams_df.to_csv(f'{data_dir}/teams.csv', index=False)
teams_df.head()""",

    """# Define seasons to collect data for
seasons = ['2022-23', '2023-24']""",

    """# Function to get game logs for a season
def get_game_logs(season):
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00'  # NBA
        )
        games = gamefinder.get_data_frames()[0]
        return games
    except Exception as e:
        print(f"Error getting game logs for season {season}: {str(e)}")
        return None""",

    """# Collect and save game data for each season
for season in tqdm(seasons, desc="Collecting season data"):
    games_df = get_game_logs(season)
    if games_df is not None:
        games_df.to_csv(f'{data_dir}/games_{season}.csv', index=False)
        print(f"Saved {len(games_df)} games for season {season}")
    time.sleep(1)  # Rate limiting""",

    """# Function to get team game logs
def get_team_game_logs(team_id, season):
    try:
        team_games = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season
        )
        return team_games.get_data_frames()[0]
    except Exception as e:
        print(f"Error getting team game logs for team {team_id} season {season}: {str(e)}")
        return None""",

    """# Collect and save team game logs for each team and season
for season in tqdm(seasons, desc="Collecting team data"):
    for _, team in teams_df.iterrows():
        team_games = get_team_game_logs(team['id'], season)
        if team_games is not None:
            team_games.to_csv(f'{data_dir}/team_games_{team["abbreviation"]}_{season}.csv', index=False)
            print(f"Saved {len(team_games)} games for team {team['abbreviation']} season {season}")
        time.sleep(0.6)  # Rate limiting"""
]

for code in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(code))

# Write the notebook to a file
with open('notebooks/01_data_collection.ipynb', 'w') as f:
    nbf.write(nb, f) 