RAW_DATA_DIR = 'data/raw/'
INTERMIDIATE_DATA_DIR = 'data/intermediate/'
PROCESSED_DATA_DIR = 'data/processed/'
IGNITION_RISK_PREDICTIONS_DIR = 'data/predictions/'
LAT_LON_BINS_FILE = 'data/processed/spatial/california_geospatial_bins.csv'
TOTAL_FIRES_PER_BIN_FILE = 'data/processed/fire/1992-2015_california_total_fires_per_bin.csv'
DATA_TRANSFORMATION_DIR = 'data_functions/'
TRAINED_MODEL = 'trained_models/parallel_RNN.h5'
WEATHER_HEATMAPS_DIR = 'map_pages/'

WEATHER_DATA_COLUMN_NAMES = [
    'date',
    'lat',
    'lon',
    'mean_air_2m',
    'apcp',
    'mean_rhum_2m',
    'mean_dpt_2m',
    'mean_pres_sfc',
    'mean_uwnd_10m',
    'mean_vwnd_10m',
    'mean_cloud_cover',
]

# WEATHER_FEATURES_TO_SCALE = [
#     'temp',
#     'rain',
#     'humidity',
#     'dew_point',
#     'pressure',
#     'uwind',
#     'vwind',
#     'cloud_cover',
# ]

FEATURES_TO_SCALE = [
    'lat',
    'lon',
    'month',
    'apcp',
    #     'crain',
    #     'veg',
    #     'ignition',
    'mean_air_2m',
    'mean_rhum_2m',
    'mean_dpt_2m',
    'mean_pres_sfc',
    'mean_uwnd_10m',
    'mean_vwnd_10m',
    #     'mean_vis',
    'mean_cloud_cover',
    'total_fires'
]

# FEATURES_TO_TRANSFORM = [
#     'apcp',
#     # 'veg',
#     'mean_pres_sfc',
#     # 'mean_vis',
#     'mean_cloud_cover',
#     # 'total_fires'
# ]

RNN_INPUT_SHAPE_PARAMETERS = {
    'history_size': 8,         # size of past history time chunk
    'target_size': 1,          # number of timepoints to predict from each history time chunk
    'step': 1                  # number of timepoints to move the history time chunk
}
