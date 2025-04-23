import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, teamgamelog
from nba_api.stats.static import teams, players
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import requests
from requests.exceptions import RequestException
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_data_collection.log'),
        logging.StreamHandler()
    ]
)

class NBADataCollector:
    def __init__(self):
        self.seasons = ['2022-23', '2023-24']  # Add more seasons as needed
        self.data_dir = '../../data/raw'
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def get_all_teams(self):
        """Get all NBA teams"""
        try:
            return pd.DataFrame(teams.get_teams())
        except Exception as e:
            logging.error(f"Error getting teams: {str(e)}")
            raise
    
    def get_game_logs(self, season):
        """Get game logs for a specific season"""
        for attempt in range(self.max_retries):
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable='00'  # NBA
                )
                games = gamefinder.get_data_frames()[0]
                return games
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to get game logs for season {season} after {self.max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed for season {season}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
    
    def get_team_game_logs(self, team_id, season):
        """Get game logs for a specific team"""
        for attempt in range(self.max_retries):
            try:
                team_games = teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=season
                )
                return team_games.get_data_frames()[0]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to get team game logs for team {team_id} season {season} after {self.max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed for team {team_id} season {season}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
    
    def get_player_game_logs(self, player_id, season):
        """Get game logs for a specific player"""
        for attempt in range(self.max_retries):
            try:
                player_games = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                return player_games.get_data_frames()[0]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to get player game logs for player {player_id} season {season} after {self.max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed for player {player_id} season {season}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
    
    def collect_all_data(self):
        """Collect all necessary data"""
        logging.info("Starting data collection...")
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Get all teams
            teams_df = self.get_all_teams()
            teams_df.to_csv(f'{self.data_dir}/teams.csv', index=False)
            logging.info("Teams data collected")
            
            # Get game logs for each season
            for season in tqdm(self.seasons, desc="Collecting season data"):
                try:
                    # Get all games for the season
                    games_df = self.get_game_logs(season)
                    games_df.to_csv(f'{self.data_dir}/games_{season}.csv', index=False)
                    
                    # Get team game logs
                    for _, team in teams_df.iterrows():
                        try:
                            team_games = self.get_team_game_logs(team['id'], season)
                            team_games.to_csv(f'{self.data_dir}/team_games_{team["abbreviation"]}_{season}.csv', index=False)
                            time.sleep(0.6)  # Rate limiting
                        except Exception as e:
                            logging.error(f"Error collecting data for team {team['abbreviation']} season {season}: {str(e)}")
                            continue
                    
                    logging.info(f"Season {season} data collected")
                except Exception as e:
                    logging.error(f"Error collecting data for season {season}: {str(e)}")
                    continue
            
            logging.info("Data collection complete!")
        except Exception as e:
            logging.error(f"Fatal error during data collection: {str(e)}")
            raise

if __name__ == "__main__":
    collector = NBADataCollector()
    collector.collect_all_data() 