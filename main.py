import requests
import yaml
import pandas as pd
import os
from tqdm import tqdm
import datetime
import numpy as np
from sqlalchemy import create_engine
import hashlib


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
                df = pd.read_csv(f'data/{country}_{div}_{season}.csv',
                                 encoding='windows-1252',
                                 skip_blank_lines=True)
                season_column = [season] * len(df.index)
                season_idx_column = [idx + 1] * len(df.index)
                df['Season'] = season_column
                df['Season_idx'] = season_idx_column
                df.to_csv(f'data/{country}_{div}_{season}.csv', sep=',', index=False)


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


def _upload_data(data: pd.DataFrame) -> None:
    """
    Uploads data to the database.

    Args:
        data: Data to be uploaded into the database.

    """
    print('\nUploading data...')
    engine = create_engine('mysql://root:root@localhost:3306/football')
    data.to_sql(name="data", con=engine, if_exists="replace", index=False)


def transform_data():
    """
    Transforms data, tidying it up and performing operations such as column renaming. Also calculates surprises and
    favours. Data is then saved to file.

    """
    print("\nTransforming data...")
    df = pd.read_csv('merged_data.csv', encoding='windows-1252', low_memory=False)
    try:
        df.drop('Unnamed: 105', axis=1, inplace=True)
    except KeyError:
        print("No Unnamed: 105 column found")
    b365_loc: int = df.columns.get_loc('B365A')

    df.insert(loc=b365_loc+1, column='B365_favour', value='')
    df.insert(loc=b365_loc+1, column='B365_surprised', value=0)
    df.insert(loc=b365_loc+6, column='bet_and_win_favour', value='')
    df.insert(loc=b365_loc+6, column='bet_and_win_surprised', value=0)
    df.insert(loc=b365_loc+11, column='interwetten_favour', value='')
    df.insert(loc=b365_loc+11, column='interwetten_surprised', value=0)
    df.insert(loc=b365_loc+16, column='ladbrokes_favour', value="")
    df.insert(loc=b365_loc+16, column='ladbrokes_surprised', value=0)
    df.insert(loc=b365_loc+21, column='pinnacle_favour', value='')
    df.insert(loc=b365_loc+21, column='pinnacle_surprised', value=0)
    df.insert(loc=b365_loc+26, column='william_hill_favour', value='')
    df.insert(loc=b365_loc+26, column='william_hill_surprised', value=0)
    df.insert(loc=b365_loc+31, column='stan_james_favour', value='')
    df.insert(loc=b365_loc+31, column='stan_james_surprised', value=0)
    df.insert(loc=b365_loc+36, column='vc_bet_favour', value='')
    df.insert(loc=b365_loc+36, column='vc_bet_surprised', value=0)

    b365_loc: int = df.columns.get_loc('B365A')
    df.insert(loc=b365_loc-2, column='guest_won', value=0)
    df.insert(loc=b365_loc-2, column='guest_lost', value=0)
    df.insert(loc=b365_loc-2, column='draw', value=0)
    df.insert(loc=b365_loc-2, column='home_won', value=0)
    df.insert(loc=b365_loc-2, column='home_lost', value=0)

    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        match row['FTR']:
            case 'H':
                df.loc[index, 'FTR'] = row['HomeTeam']
                df.loc[index, 'guest_lost'] = 1
                df.loc[index, 'home_won'] = 1
            case 'A':
                df.loc[index, 'FTR'] = row['AwayTeam']
                df.loc[index, 'guest_won'] = 1
                df.loc[index, 'home_lost'] = 1
            case 'D':
                df.loc[index, 'FTR'] = 'Draw'
                df.loc[index, 'draw'] = 1
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

        match pd.to_numeric(row[['BWH', 'BWD', 'BWA']]).idxmin():
            case 'BWH':
                df.loc[index, 'bet_and_win_favour'] = row['HomeTeam']
            case 'BWD':
                df.loc[index, 'bet_and_win_favour'] = 'Draw'
            case 'BWA':
                df.loc[index, 'bet_and_win_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'bet_and_win_favour'] = np.nan

        match pd.to_numeric(row[['IWH', 'IWD', 'IWA']]).idxmin():
            case 'IWH':
                df.loc[index, 'interwetten_favour'] = row['HomeTeam']
            case 'IWD':
                df.loc[index, 'interwetten_favour'] = 'Draw'
            case 'IWA':
                df.loc[index, 'interwetten_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'interwetten_favour'] = np.nan

        match pd.to_numeric(row[['LBH', 'LBD', 'LBA']]).idxmin():
            case 'LBH':
                df.loc[index, 'ladbrokes_favour'] = row['HomeTeam']
            case 'LBD':
                df.loc[index, 'ladbrokes_favour'] = 'Draw'
            case 'LBA':
                df.loc[index, 'ladbrokes_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'ladbrokes_favour'] = np.nan

        match pd.to_numeric(row[['PSH', 'PSD', 'PSA']]).idxmin():
            case 'PSH':
                df.loc[index, 'pinnacle_favour'] = row['HomeTeam']
            case 'PSD':
                df.loc[index, 'pinnacle_favour'] = 'Draw'
            case 'PSA':
                df.loc[index, 'pinnacle_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'pinnacle_favour'] = np.nan

        match pd.to_numeric(row[['WHH', 'WHD', 'WHA']]).idxmin():
            case 'WHH':
                df.loc[index, 'william_hill_favour'] = row['HomeTeam']
            case 'WHD':
                df.loc[index, 'william_hill_favour'] = 'Draw'
            case 'WHA':
                df.loc[index, 'william_hill_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'william_hill_favour'] = np.nan

        match pd.to_numeric(row[['SJH', 'SJD', 'SJA']]).idxmin():
            case 'SJH':
                df.loc[index, 'stan_james_favour'] = row['HomeTeam']
            case 'SJD':
                df.loc[index, 'stan_james_favour'] = 'Draw'
            case 'SJA':
                df.loc[index, 'stan_james_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'stan_james_favour'] = np.nan

        match pd.to_numeric(row[['VCH', 'VCD', 'VCA']]).idxmin():
            case 'VCH':
                df.loc[index, 'vc_bet_favour'] = row['HomeTeam']
            case 'VCD':
                df.loc[index, 'vc_bet_favour'] = 'Draw'
            case 'VCA':
                df.loc[index, 'vc_bet_favour'] = row['AwayTeam']
            case _:
                df.loc[index, 'vc_bet_favour'] = np.nan

        if pd.isna(df.loc[index, 'B365_favour']):
            df.loc[index, 'B365_surprised'] = np.nan
        elif df.loc[index, 'B365_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'B365_favour'] == row['AwayTeam']:
                    df.loc[index, 'B365_surprised'] = round(row['B365H'] - row['B365A'], 2)
                else:
                    df.loc[index, 'B365_surprised'] = round(row['B365H'] - row['B365D'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'B365_favour'] == row['HomeTeam']:
                    df.loc[index, 'B365_surprised'] = round(row['B365D'] - row['B365H'], 2)
                else:
                    df.loc[index, 'B365_surprised'] = round(row['B365D'] - row['B365A'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'B365_favour'] == row['HomeTeam']:
                    df.loc[index, 'B365_surprised'] = round(row['B365A'] - row['B365H'], 2)
                else:
                    df.loc[index, 'B365_surprised'] = round(row['B365A'] - row['B365D'], 2)

        if pd.isna(df.loc[index, 'bet_and_win_favour']):
            df.loc[index, 'bet_and_win_surprised'] = np.nan
        elif df.loc[index, 'bet_and_win_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'bet_and_win_favour'] == row['AwayTeam']:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWH'] - row['BWA'], 2)
                else:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWH'] - row['BWD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'bet_and_win_favour'] == row['HomeTeam']:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWD'] - row['BWH'], 2)
                else:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWD'] - row['BWA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'bet_and_win_favour'] == row['HomeTeam']:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWA'] - row['BWH'], 2)
                else:
                    df.loc[index, 'bet_and_win_surprised'] = round(row['BWA'] - row['BWD'], 2)

        if pd.isna(df.loc[index, 'interwetten_favour']):
            df.loc[index, 'interwetten_surprised'] = np.nan
        elif df.loc[index, 'interwetten_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'interwetten_favour'] == row['AwayTeam']:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWH'] - row['IWA'], 2)
                else:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWH'] - row['IWD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'interwetten_favour'] == row['HomeTeam']:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWD'] - row['IWH'], 2)
                else:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWD'] - row['IWA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'interwetten_favour'] == row['HomeTeam']:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWA'] - row['IWH'], 2)
                else:
                    df.loc[index, 'interwetten_surprised'] = round(row['IWA'] - row['IWD'], 2)

        if pd.isna(df.loc[index, 'ladbrokes_favour']):
            df.loc[index, 'ladbrokes_surprised'] = np.nan
        elif df.loc[index, 'ladbrokes_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'ladbrokes_favour'] == row['AwayTeam']:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBH'] - row['LBA'], 2)
                else:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBH'] - row['LBD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'ladbrokes_favour'] == row['HomeTeam']:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBD'] - row['LBH'], 2)
                else:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBD'] - row['LBA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'ladbrokes_favour'] == row['HomeTeam']:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBA'] - row['LBH'], 2)
                else:
                    df.loc[index, 'ladbrokes_surprised'] = round(row['LBA'] - row['LBD'], 2)

        if pd.isna(df.loc[index, 'pinnacle_favour']):
            df.loc[index, 'pinnacle_surprised'] = np.nan
        elif df.loc[index, 'pinnacle_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'pinnacle_favour'] == row['AwayTeam']:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSH'] - row['PSA'], 2)
                else:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSH'] - row['PSD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'pinnacle_favour'] == row['HomeTeam']:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSD'] - row['PSH'], 2)
                else:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSD'] - row['PSA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'pinnacle_favour'] == row['HomeTeam']:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSA'] - row['PSH'], 2)
                else:
                    df.loc[index, 'pinnacle_surprised'] = round(row['PSA'] - row['PSD'], 2)

        if pd.isna(df.loc[index, 'william_hill_favour']):
            df.loc[index, 'william_hill_surprised'] = np.nan
        elif df.loc[index, 'william_hill_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'william_hill_favour'] == row['AwayTeam']:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHH'] - row['WHA'], 2)
                else:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHH'] - row['WHD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'william_hill_favour'] == row['HomeTeam']:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHD'] - row['WHH'], 2)
                else:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHD'] - row['WHA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'william_hill_favour'] == row['HomeTeam']:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHA'] - row['WHH'], 2)
                else:
                    df.loc[index, 'william_hill_surprised'] = round(row['WHA'] - row['WHD'], 2)

        if pd.isna(df.loc[index, 'stan_james_favour']):
            df.loc[index, 'stan_james_surprised'] = np.nan
        elif df.loc[index, 'stan_james_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'stan_james_favour'] == row['AwayTeam']:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJH'] - row['SJA'], 2)
                else:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJH'] - row['SJD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'stan_james_favour'] == row['HomeTeam']:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJD'] - row['SJH'], 2)
                else:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJD'] - row['SJA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'stan_james_favour'] == row['HomeTeam']:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJA'] - row['SJH'], 2)
                else:
                    df.loc[index, 'stan_james_surprised'] = round(row['SJA'] - row['SJD'], 2)

        if pd.isna(df.loc[index, 'vc_bet_favour']):
            df.loc[index, 'vc_bet_surprised'] = np.nan
        elif df.loc[index, 'vc_bet_favour'] != df.loc[index, 'FTR']:
            if df.loc[index, 'home_won'] == 1:
                if df.loc[index, 'vc_bet_favour'] == row['AwayTeam']:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCH'] - row['VCA'], 2)
                else:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCH'] - row['VCD'], 2)
            elif df.loc[index, 'draw'] == 1:
                if df.loc[index, 'vc_bet_favour'] == row['HomeTeam']:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCD'] - row['VCH'], 2)
                else:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCD'] - row['VCA'], 2)
            elif df.loc[index, 'guest_won'] == 1:
                if df.loc[index, 'vc_bet_favour'] == row['HomeTeam']:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCA'] - row['VCH'], 2)
                else:
                    df.loc[index, 'vc_bet_surprised'] = round(row['VCA'] - row['VCD'], 2)


    df.rename(columns={
        'FTHG': 'fulltime_home_goals',
        'FTAG': 'fulltime_away_goals',
        'FTR': 'fulltime_match_result',
        'HTHG': 'halftime_home_goals',
        'HTAG': 'halftime_away_goals',
        'HTR': 'halftime_match_result',
        'HC': 'home_team_corners',
        'AC': 'away_team_corners',
        'HY': 'home_team_yellow_cards',
        'AY': 'away_team_yellow_cards',
        'HR': 'home_team_red_cards',
        'AR': 'away_team_red_cards',
        'HF': 'home_team_fouls_committed',
        'AF': 'away_team_fouls_committed',
        'HS': 'home_team_shots',
        'AS': 'away_team_shots',
        'HST': 'home_team_shots_on_target',
        'AST': 'away_team_shots_on_target'
    }, inplace=True)

    columns = ['B365_surprised', 'bet_and_win_surprised', 'interwetten_surprised', 'ladbrokes_surprised',
               'pinnacle_surprised', 'william_hill_surprised', 'stan_james_surprised', 'vc_bet_surprised']

    for column in columns:
        df[column] = round((df[column] - df[column].min()) / (df[column].max() - df[column].min()), 2)

    _create_id(df)
    df.dropna(subset=['Div'], inplace=True)
    _upload_data(df)
    df.to_csv('merged_data_clean.csv', index=False)
    pass


def _create_id(dataframe: pd.DataFrame) -> None:
    """
    Adds two new columns to dataframe: ID and checksum, both of which are hashes of certain columns. This is performed
    to check for duplicates, which shouldn't happen.

    Args:
        dataframe: Main dataframe containing all columns.

    """
    id_column_loc: int = 0
    dataframe.insert(loc=id_column_loc, column='ID', value='')
    dataframe['ID'] = dataframe['HomeTeam'].astype(str) + dataframe['AwayTeam'].astype(str) + dataframe['Date'].astype(str)
    dataframe['ID'] = dataframe['ID'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    dataframe['Checksum'] = dataframe['Div'].astype(str) + dataframe['Date'].astype(str) + dataframe['HomeTeam'].astype(str) + dataframe['AwayTeam'].astype(str) + dataframe['fulltime_home_goals'].astype(str) + dataframe['fulltime_away_goals'].astype(str) + dataframe['fulltime_match_result'].astype(str) + dataframe['halftime_home_goals'].astype(str) + dataframe['halftime_away_goals'].astype(str) + dataframe['halftime_match_result'].astype(str) + dataframe['B365H'].astype(str) + dataframe['B365D'].astype(str) + dataframe['B365A'].astype(str) + dataframe['B365_favour'].astype(str) + dataframe['BWH'].astype(str) + dataframe['BWD'].astype(str) + dataframe['BWA'].astype(str) + dataframe['IWH'].astype(str) + dataframe['IWD'].astype(str) + dataframe['IWA'].astype(str) + dataframe['LBH'].astype(str) + dataframe['LBD'].astype(str) + dataframe['LBA'].astype(str) + dataframe['PSH'].astype(str) + dataframe['PSD'].astype(str) + dataframe['PSA'].astype(str) + dataframe['WHH'].astype(str) + dataframe['WHD'].astype(str) + dataframe['WHA'].astype(str) + dataframe['SJH'].astype(str) + dataframe['SJD'].astype(str) + dataframe['SJA'].astype(str) + dataframe['VCH'].astype(str) + dataframe['VCD'].astype(str) + dataframe['VCD'].astype(str) + dataframe['Bb1X2'].astype(str) + dataframe['BbMxH'].astype(str) + dataframe['BbAvH'].astype(str) + dataframe['BbMxD'].astype(str) + dataframe['BbAvD'].astype(str) + dataframe['BbMxA'].astype(str) + dataframe['BbAvA'].astype(str) + dataframe['BbOU'].astype(str) + dataframe['BbMx>2.5'].astype(str) + dataframe['BbAv>2.5'].astype(str) + dataframe['BbMx<2.5'].astype(str) + dataframe['BbAv<2.5'].astype(str) + dataframe['BbAH'].astype(str) + dataframe['BbAHh'].astype(str) + dataframe['BbMxAHH'].astype(str) + dataframe['BbAvAHH'].astype(str) + dataframe['BbMxAHH'].astype(str) + dataframe['BbMxAHA'].astype(str) + dataframe['BbAvAHA'].astype(str) + dataframe['PSCD'].astype(str) + dataframe['PSCA'].astype(str) + dataframe['Season'].astype(str) + dataframe['Season_idx'].astype(str) + dataframe['home_team_shots'].astype(str) + dataframe['away_team_shots'].astype(str) + dataframe['home_team_shots_on_target'].astype(str) + dataframe['away_team_shots_on_target'].astype(str) + dataframe['HFKC'].astype(str) + dataframe['AFKC'].astype(str) + dataframe['home_team_corners'].astype(str) + dataframe['away_team_corners'].astype(str) + dataframe['home_team_yellow_cards'].astype(str) + dataframe['away_team_yellow_cards'].astype(str) + dataframe['home_team_red_cards'].astype(str) + dataframe['away_team_red_cards'].astype(str) + dataframe['Time'].astype(str) + dataframe['home_team_fouls_committed'].astype(str) + dataframe['away_team_fouls_committed'].astype(str) + dataframe['MaxH'].astype(str) + dataframe['MaxD'].astype(str) + dataframe['MaxH'].astype(str) + dataframe['MaxD'].astype(str) + dataframe['MaxA'].astype(str) + dataframe['AvgH'].astype(str) + dataframe['AvgD'].astype(str) + dataframe['AvgA'].astype(str) + dataframe['B365>2.5'].astype(str) + dataframe['B365<2.5'].astype(str) + dataframe['P>2.5'].astype(str) + dataframe['P<2.5'].astype(str) + dataframe['Max>2.5'].astype(str) + dataframe['Max<2.5'].astype(str) + dataframe['Avg>2.5'].astype(str) + dataframe['Avg<2.5'].astype(str) + dataframe['AHh'].astype(str) + dataframe['B365AHH'].astype(str) + dataframe['B365AHA'].astype(str) + dataframe['PAHH'].astype(str) + dataframe['PAHA'].astype(str) + dataframe['MaxAHH'].astype(str) + dataframe['MaxAHA'].astype(str) + dataframe['AvgAHH'].astype(str) + dataframe['AvgAHA'].astype(str) + dataframe['B365CH'].astype(str) + dataframe['B365CD'].astype(str) + dataframe['B365CA'].astype(str) + dataframe['BWCH'].astype(str) + dataframe['BWCD'].astype(str) + dataframe['BWCA'].astype(str) + dataframe['IWCH'].astype(str) + dataframe['IWCD'].astype(str) + dataframe['IWCA'].astype(str) + dataframe['WHCH'].astype(str) + dataframe['WHCD'].astype(str) + dataframe['WHCA'].astype(str) + dataframe['VCCH'].astype(str) + dataframe['VCCD'].astype(str) + dataframe['VCCD'].astype(str) + dataframe['MaxCH'].astype(str) + dataframe['MaxCD'].astype(str) + dataframe['AvgCH'].astype(str) + dataframe['AvgCA'].astype(str) + dataframe['B365C>2.5'].astype(str) + dataframe['B365C<2.5'].astype(str) + dataframe['PC>2.5'].astype(str) + dataframe['PC<2.5'].astype(str) + dataframe['MaxC>2.5'].astype(str) + dataframe['MaxC<2.5'].astype(str) + dataframe['AvgC>2.5'].astype(str) + dataframe['AvgC<2.5'].astype(str) + dataframe['AHCh'].astype(str) + dataframe['B365CAHH'].astype(str) + dataframe['B365CAHA'].astype(str) + dataframe['PCAHH'].astype(str) + dataframe['PCAHA'].astype(str) + dataframe['MaxCAHH'].astype(str) + dataframe['MaxCAHA'].astype(str) + dataframe['AvgCAHH'].astype(str) + dataframe['AvgCAHA'].astype(str) + dataframe['Referee'].astype(str)
    dataframe['Checksum'] = dataframe['Checksum'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())


if __name__ == '__main__':
    gather_seasons()
    gather_data()
    merge_data()
    transform_data()
    print("Done!")
    pass
