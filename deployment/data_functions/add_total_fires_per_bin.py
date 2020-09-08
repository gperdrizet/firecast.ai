import pandas as pd


def add_fires_per_bin(parsed_weather_data_file: str, fires_per_bin_file: str) -> 'DataFrame':
    '''Takes parsed weather data and parquet file containing total number of 
    wildfires per geospatial bin. Joins on lat and lon'''

    weather_data = pd.read_parquet(parsed_weather_data_file)
    fires_per_bin = pd.read_csv(fires_per_bin_file)

    # fires_per_bin.reset_index(inplace=True)
    # weather_data.reset_index(inplace=True)

    # weather_data = weather_data.round({'lat': 2, 'lon': 2})
    # fires_per_bin = fires_per_bin.round({'lat': 2, 'lon': 2})

    fires_per_bin.set_index(['lat', 'lon'], inplace=True)
    # weather_data.set_index(['lat', 'lon'], inplace=True)

    one_day = weather_data[weather_data['date'] == '2020-08-30']
    one_day.to_csv('data/processed/fire/1992-2015_bins.csv')

    weather_data['total_fires'] = weather_data.apply(
        lambda x: fires_per_bin.loc[(x['lat'], x['lon'])], axis=1)

    return weather_data
