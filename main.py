import requests
import yaml
import pandas as pd
import os
from tqdm import tqdm
import datetime


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
    print("Gathering data...")
    countries = config['div_urls'].keys()
    for country in tqdm(countries):
        for div in config['div_urls'][country].values():
            for season in seasons:

                url = f'https://www.football-data.co.uk/mmz4281/{season}/{div}.csv'
                r = requests.get(url, allow_redirects=True)

                open(f'data/{country}_{div}_{season}.csv', 'wb').write(r.content)
                df = pd.read_csv(f'data/{country}_{div}_{season}.csv', encoding='windows-1252')
                season_column = [season] * len(df.index)
                df['Season'] = season_column
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


def clean_data():
    print("\nCleaning data...")
    df = pd.read_csv('merged_data.csv', encoding='windows-1252', low_memory=False)
    try:
        df.drop('Unnamed: 105', axis=1, inplace=True)
    except KeyError:
        print("No Unnamed: 105 column found")
    df.to_csv('merged_data_clean.csv', index=False)
    pass


if __name__ == '__main__':
    # gather_seasons()
    # gather_data()
    # merge_data()
    clean_data()
    print("Done!")
    pass
