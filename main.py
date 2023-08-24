import requests
import yaml

try:
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
except FileNotFoundError:
    print('config.yaml not found')
    exit(1)

# leagues_urls = {
#     'England': 'https://www.football-data.co.uk/mmz4281/'
# }
#
# # Tylko pierwsze 2
# england_divs_urls = {
#     'Premier League': 'E0',
#     'Championship': 'E1',
# }
#
# scotland_divs_urls = {
#     'Premier League': 'SC0',
#     'Division 1': 'SC1',
# }
#
# germany_divs_urls = {
#     'Bundesliga 1': 'D1',
#     'Bundesliga 2': 'D2',
# }
#
# italy_divs_urls = {
#     'Serie A': 'I1',
#     'Serie B': 'I2',
# }
#
# spain_divs_urls = {
#     'La Liga Primera': 'SP1',
#     'La Liga Segunda': 'SP2',
# }
#
# france_divs_urls = {
#     'Le Championnat': 'F1',
#     'Division 2': 'F2',
# }
#
# netherlands_divs_urls = {
#     'Eredivisie': 'N1',
# }
#
# belgium_divs_urls = {
#     'Jupiler League': 'B1',
# }
#
# portugal_divs_urls = {
#     'Liga I': 'P1',
# }
#
# turkey_divs_urls = {
#     'Futbol Ligi 1': 'T1',
# }
#
# greece_divs_urls = {
#     'Ethniki Katigoria': 'G1',
# }



def get_main_csv():
    url = 'https://www.football-data.co.uk/mmz4281/2324/Latest_Results.csv'
    r = requests.get(url, allow_redirects=True)

    open('main_csv.csv', 'wb').write(r.content)


if __name__ == '__main__':
    print(config)
    pass
