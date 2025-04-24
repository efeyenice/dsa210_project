# NBA Game Outcome Prediction

This project uses Jupyter notebooks to collect and analyze NBA game data for predicting game outcomes.

## Project Structure

```
project/
├── data/               # Directory for storing collected data
├── env/                # Environment files
│   ├── environment.yml # Conda environment file
│   └── requirements.txt # Pip requirements file
├── notebooks/          # Jupyter notebooks for data collection and analysis
└── README.md           # Project documentation
```

## Environment Setup

### Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
conda env create -f env/environment.yml
conda activate nba_project
```

### Using Pip

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r env/requirements.txt
```

## Running the Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/01_data_collection.ipynb` and run the cells in sequence.

## Notebooks

1. `01_data_collection.ipynb`: Collects NBA game data from the NBA API and saves it to CSV files
   - Collects team data
   - Collects game data for each season
   - Collects team-specific game logs

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- nba_api
- tqdm

## Project Overview
Over the next three months, I will analyze NBA game data to predict whether a team will win or lose based on key statistics. By leveraging machine learning techniques, I aim to identify which factors contribute most to a team's success. This project will involve data collection, feature engineering, exploratory data analysis, and predictive modeling.

## Motivation
I have followed NBA for years, discussing and analyzing games with friends. Predicting outcomes adds another layer of engagement. Also, basketball is my passion. I have been playing this sport for a long time and analyzing the key factors that contribute to a team's success will be fun and interesting for me.

## Objectives
- Analyze Key Game Factors: Identify the relationship between team and player statistics and game outcomes.
- Feature Selection & Engineering: Determine the most influential statistics and create additional features that improve prediction accuracy.
- Develop and Evaluate Models: Apply machine learning models to predict game outcomes and assess their performance.

## Dataset
The dataset consists of historical NBA game records, player statistics, and contextual factors. Data will be collected from the following sources:
- NBA API / Basketball-Reference: Provides detailed game statistics, team records, and historical match data.
- Kaggle Datasets: Contains historical NBA box scores and related analytics datasets.

## Analysis Plan
1. Data Collection & Cleaning: Gather data from APIs and datasets. Handle missing values and ensure data consistency.
2. Exploratory Data Analysis: Analyze correlations between different statistics and game outcomes. Compare team performance trends over time. Investigate the impact of factors like home-court advantage and injuries.
3. Feature Engineering: Create rolling averages for last 5/10 games. Adjust for strength of schedule. Develop impact ratings for key players. Model injury effects on performance.
4. Predictive Modeling: Use models like Random Forest, XGBoost. Identify key features that contribute most to game outcomes. Compare predictions against actual results.

## Hypothesis
- Null Hypothesis (H₀): There is no significant relationship between various factors and game outcomes. In other words, variations in factors do not significantly affect whether a team wins or loses a game.
- Alternative Hypothesis (Hₐ): There is a significant relationship between various factors and game outcomes. Specifically, differences in factors are associated with variations in a team's probability of winning.