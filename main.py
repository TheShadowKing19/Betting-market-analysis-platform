import requests
import yaml
import pandas as pd
import os
from tqdm import tqdm
import datetime
import numpy as np
from bs4 import BeautifulSoup


try:
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
except FileNotFoundError:
    print('config.yaml not found')
    exit(1)


def gather_seasons() -> list[str]:
    """
    Prepares a list of seasons to gather data from, counting from the current season to 10 seasons back.

    Returns:
        List of seasons in the format 'YYYY' e.g. ['2122', '2223', '2324', ...]

    """
    seasons = []
    current_year = datetime.datetime.today().strftime('%y')
    current_season = f'{current_year}{int(current_year) + 1}'
    seasons.append(current_season)
    for _ in range(9):
        current_season = f'{int(current_season[:2]) - 1}{int(current_season[2:]) - 1}'
        seasons.append(current_season)
    return seasons


def gather_data() -> None:
    """
    Gathers data for every country, division and season from football-data.co.uk and saves it to the "data" folder.

    """
    if not os.path.exists('data'):
        os.mkdir('data')

    seasons = gather_seasons()
    seasons.reverse()
    print("Gathering data...")
    countries = config['div_urls'].keys()
    for country in tqdm(countries):
        for div in config['div_urls'][country].values():
            for idx, season in enumerate(seasons):

                url = f'https://www.football-data.co.uk/mmz4281/{season}/{div}.csv'
                r = requests.get(url, allow_redirects=True)

                open(f'data/{country}_{div}_{season}.csv', 'wb').write(r.content)
                df = pd.read_csv(f'data/{country}_{div}_{season}.csv', encoding='windows-1252')
                season_column = [season] * len(df.index)
                season_idx_column = [idx + 1] * len(df.index)
                df['Season'] = season_column
                df['Season_idx'] = season_idx_column
                df.to_csv(f'data/{country}_{div}_{season}.csv', sep=',', index=False)


def _get_current_team_value(season: str):
    pass


def merge_data() -> None:
    """
    Merges all the data from the "data" folder into a single csv file.

    """
    files = os.listdir('data')
    df = pd.DataFrame()
    print("\nMerging data...")
    for file in tqdm(files):
        df = pd.concat([df, pd.read_csv(f'data/{file}', encoding='windows-1252')],
                       axis=0,
                       ignore_index=True)
    df.to_csv('merged_data.csv', index=False)
    pass


def transform_data():
    print("\nTransforming data...")
    df = pd.read_csv('merged_data.csv', encoding='windows-1252', low_memory=False)
    try:
        df.drop('Unnamed: 105', axis=1, inplace=True)
    except KeyError:
        print("No Unnamed: 105 column found")
    b365_loc: int = df.columns.get_loc('B365A')
    df.insert(loc=b365_loc+1, column='B365_favour', value='')
    for index, row in df.iterrows():
        match row['FTR']:
            case 'H':
                df.loc[index, 'FTR'] = row['HomeTeam']
            case 'A':
                df.loc[index, 'FTR'] = row['AwayTeam']
            case 'D':
                df.loc[index, 'FTR'] = 'Draw'
            case _:
                df.loc[index, 'FTR'] = np.nan
        match row['HTR']:
            case 'H':
                df.loc[index, 'HTR'] = row['HomeTeam']
            case 'A':
                df.loc[index, 'HTR'] = row['AwayTeam']
            case 'D':
                df.loc[index, 'HTR'] = 'Draw'
            case _:
                df.loc[index, 'HTR'] = np.nan
        match pd.to_numeric(row[['B365H', 'B365D', 'B365A']]).idxmin():
            case 'B365H':
                df.loc[index, 'B365_favour'] = row['HomeTeam']
            case 'B365D':
                df.loc[index, 'B365_favour'] = 'Draw'
            case 'B365A':
                df.loc[index, 'B365_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'B365_favour'] = np.nan

    df.rename(columns={
        'FTHG': 'fulltime_home_goals',
        'FTAG': 'fulltime_away_goals',
        'FTR': 'fulltime_match_result',
        'HTHG': 'halftime_home_goals',
        'HTAG': 'halftime_away_goals',
        'HTR': 'halftime_match_result',
    }, inplace=True)
    df.to_csv('merged_data_clean.csv', index=False)
    pass


if __name__ == '__main__':
    gather_seasons()
    gather_data()
    merge_data()
    transform_data()
    print("Done!")
    pass
