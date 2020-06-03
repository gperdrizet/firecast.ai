import requests
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile


def get_fire_data(url: str, path: str, file_name: str) -> 'DataFrame':
    zip_archive = f'{path}{file_name}.zip'
    stream = requests.get(url, stream=True)

    if os.path.isfile(zip_archive):
        print("Zip file exist")

    else:
        with open(zip_archive, 'wb') as output:
            for chunk in stream.iter_content(chunk_size=128):
                output.write(chunk)

    with ZipFile(zip_archive) as z:
        sql_file = f'{path}{file_name}.sqlite'
        with open(sql_file, 'wb') as f:
            f.write(z.read('Data/FPA_FOD_20170508.sqlite'))

    fire_sql_conn = sqlite3.connect(sql_file)
    fires = pd.read_sql_query("SELECT * from fires", fire_sql_conn)

    california_fires = fires[fires['STATE'] == 'CA']
    california_fires = california_fires[[
        'LATITUDE', 'LONGITUDE', 'DISCOVERY_DATE']]

    california_fires.columns = ['lat', 'lon', 'time']
    california_fires['time'] = pd.to_datetime(
        california_fires['time'], unit='D', origin='julian')

    california_fires.sort_values(by=['time'])

    california_fires.drop_duplicates(keep=False, inplace=True)

    california_fires = california_fires.astype(
        {'lat': 'float32', 'lon': 'float32'})

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.scatter(x=california_fires['lon'],
                y=california_fires['lat'], color='darkred', s=0.5)

    plt.axis('equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('California wildfires 1992 - 2015')

    plt.savefig(
        '../project_info/figures/california_fires_scatterplot.png', bbox_inches='tight')

    return california_fires
