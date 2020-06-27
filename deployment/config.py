RAW_DATA_DIR = 'data/raw/'
INTERMIDIATE_DATA_DIR = 'data/intermediate/'
PROCESSED_DATA_DIR = 'data/processed/'
IGNITION_RISK_PREDICTIONS_DIR = 'data/predictions/'
LAT_LON_BINS_FILE = 'data/intermediate/spatial/california_geospatial_bins.csv'

DATA_TRANSFORMATION_DIR = 'data_functions/'

WEATHER_DATA_COLUMN_NAMES = [
    'date',
    'lat',
    'lon',
    'temp',
    'rain',
    'humidity',
    'dew_point',
    'pressure',
    'uwind',
    'vwind',
    'cloud_cover',
]

WEATHER_FEATURES_TO_SCALE = [
    'temp',
    'rain',
    'humidity',
    'dew_point',
    'pressure',
    'uwind',
    'vwind',
    'cloud_cover',
]

LSTM_INPUT_SHAPE_PARAMETERS = {
    'history_size': 5,         # size of past history time chunk
    'target_size': 1,          # number of timepoints to predict from each history time chunk
    'step': 1                  # number of timepoints to move the history time chunk
}

LSTM_HYPERPARAMETERS = {
    'batch_size': 1,
    'lstm_units': 1,
    'variational_dropout': 0.24,
    'hidden_units': 410,
    'hidden_l1_lambda': 0.1,
    'output_bias': -2.68
}
