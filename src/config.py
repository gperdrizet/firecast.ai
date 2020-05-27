RAW_WEATHER_DATA_DIR = '../data/raw/'
INTERMIDIATE_WEATHER_DATA_DIR = '../data/intermediate/'

LAT_LON_BINS_FILE = '../data/intermediate/california_geospatial_bins.csv'

WEATHER_DATA_COLUMN_NAMES = [
    'lat',
    'lon',
    'date',
    'temp',
    'pressure',
    'humidity',
    'dew_point',
    'uwind',
    'vwind',
    'cloud_cover',
    'rain'
]

WEATHER_FEATURES_TO_SCALE = [
    'temp',
    'pressure',
    'humidity',
    'dew_point',
    'uwind',
    'vwind',
    'cloud_cover',
    'rain'
]
