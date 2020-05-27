RAW_WEATHER_DATA_DIR = '../data/raw/'
INTERMIDIATE_WEATHER_DATA_DIR = '../data/intermediate/'
PROCESSED_WEATHER_DATA_DIR = '../data/processed/'

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

LSTM_INPUT_SHAPE_PARAMETERS = {
    'history_size': 5,         # size of past history time chunk
    'target_size': 1,          # number of timepoints to predict from each history time chunk
    'step': 1                  # number of timepoints to move the history time chunk
}
