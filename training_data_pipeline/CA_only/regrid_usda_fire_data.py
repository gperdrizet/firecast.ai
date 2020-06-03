import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial


def regrid_fire_data(fire_data_file: 'str', sample_weather_data_file: 'str', california_geospatial_bins_file: 'str') -> 'DataFrame':
    fires = pd.read_parquet(fire_data_file)
    weather = pd.read_parquet(sample_weather_data_file)
    bins = weather[['lat', 'lon']]
    unique_bins = bins.drop_duplicates()
    # unique_bins.to_parquet(california_geospatial_bins_file)

    bin_array = np.column_stack([unique_bins['lon'], unique_bins['lat']])
    fire_array = np.column_stack([fires['lon'], fires['lat']])

    bin_tree = spatial.cKDTree(bin_array)
    dist, indexes = bin_tree.query(fire_array)
    indexes = pd.Series(indexes)

    fire_bins = []

    for index in indexes:
        fire_bins.append([unique_bins.iloc[index, 0],
                          unique_bins.iloc[index, 1]])

    fires[['lat', 'lon']] = fire_bins

    fires.drop_duplicates(keep=False, inplace=True)

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.scatter(x=fires['lon'], y=fires['lat'], color='darkred', s=0.5)
    plt.axis('equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Regridded California fires 1992 - 2015')
    plt.savefig(
        '../project_info/figures/regridded_california_fires_scatterplot.png', bbox_inches='tight')

    return fires
