'''varaibles and parameters for the training data collection pipeline'''

###################
# Parallelization #
###################

# Local parallelization options
CONVERT_CLEAN_PROCESSES = 3
COARSE_SPATIAL_FILTER_PROCESSES = 3
FINE_SPATIAL_FILTER_PROCESSES = 16
COLLECT_PROCESSES = 16
JOBS_PER_PROCESS = 1

# spark cluster options
SPARK_MASTER = 'spark://192.168.2.1:7077'


#####################
# NOAA weather data #
#####################

NOAA_BASE_URL = 'https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/NARR/monolevel/'
WEATHER_DATA_BASE_PATH = '/mnt/SSD/data/NOAA_weather_data/'
RAW_DATAFILE_SUBDIR = 'raw_NOAA_datafiles/'
PARSED_DATAFILE_SUBDIR = 'parsed_NOAA_datafile/'
NOAA_NARR_FILE_EXT = 'nc'

START_YEAR = 1992
END_YEAR = 2015
DATA_YEARS = range(START_YEAR, (END_YEAR + 1), 1)

DATA_TYPES = {
    'air.sfc': 'float32',  # Surface air temp
    'air.2m': 'float32',   # Air temp. at 2 meters above surface
    'apcp': 'float32',     # Accumulated precipitation
    'crain': 'bool',       # Catagorical rain at surface
    'rhum.2m': 'float32',  # Relative humidity 2 meters above surface
    'dpt.2m': 'float32',   # Dew point temp. 2 meters above surface
    'pres.sfc': 'float32', # Pressure at surface
    'uwnd.10m': 'float32', # u component of wind (+ from west) at 10 meters
    'vwnd.10m': 'float32', # v component of wind (+ from south) at 10 meters
    'veg': 'float32',      # Vegitation at surface
    'lcdc': 'float32',     # Low cloud area fraction
    'hcdc': 'float32',     # High cloud area fraction
    'mcdc': 'float32',     # Medium cloud area fraction
    'prate': 'float32',    # Precipitation rate
    'vis': 'float32'       # Visibility
}


################
# Spatial data #
################

LAT_START = 27.00
LON_START = -124.0
LAT_END = 47.50
LON_END = -66.50

SHAPEFILE = '/mnt/SSD/data/spatial_data/cb_2018_us_nation_5m.shp'
SAMPLE_WEATHER_DATA_FILE = 'air.sfc.1992.parquet'
TARGET_GEOSPATIAL_BINS_FILE = '/mnt/SSD/data/spatial_data/noaa_weather_data_geospatial_bins.parquet'
TARGET_POLYGON = 146


#############
# Fire data #
#############

FIRE_DATA_BASE_PATH = '/mnt/SSD/data/USDA_wildfire_data/'

#################
# training data #
#################

TRAINING_DATA_BASE_PATH = '/mnt/SSD/data/training_data/'

DTYPES = {
    'air.sfc': 'float32',  # Surface air temp
    'air.2m': 'float32',   # Air temp. at 2 meters above surface
    'apcp': 'float32',     # Accumulated precipitation
    'crain': 'bool',       # Catagorical rain at surface
    'rhum.2m': 'float32',  # Relative humidity 2 meters above surface
    'dpt.2m': 'float32',   # Dew point temp. 2 meters above surface
    'pres.sfc': 'float32', # Pressure at surface
    'uwnd.10m': 'float32', # u component of wind (+ from west) at 10 meters
    'vwnd.10m': 'float32', # v component of wind (+ from south) at 10 meters
    'veg': 'float32',      # Vegitation at surface
    'lcdc': 'float32',     # Low cloud area fraction
    'hcdc': 'float32',     # High cloud area fraction
    'mcdc': 'float32',     # Medium cloud area fraction
    'prate': 'float32',    # Precipitation rate
    'vis': 'float32'       # Visibility
}

FEATURES = [
    'time', # Note: daily avg. data has no time column, just day, month, year
    'date',
    'air_2m',
    'apcp',
    'rhum_2m',
    'dpt_2m',
    'pres_sfc',
    'uwnd_10m', 
    'vwnd_10m',
    'veg',
    'vis',
    'lat',
    'lon',
    'ignition'
]

#############
# Log files #
#############

GET_WEATHER_DATA_LOG = './logs/get_weather_data.log'
CONVERT_CLEAN_RAW_NOAA_DATA_LOG = './logs/convert_clean_raw_noaa_data.log'
GET_TARGET_GEOSPATIAL_BINS_LOG ='./logs/get_target_geospatial_bins.log'
FINE_GEOSPATIAL_FILTER_NOAA_WEATHER_DATA_LOG = './logs/fine_geospatial_filter_noaa_weather_data.log'
COLLECT_NOAA_WEATHER_DATA_LOG = './logs/collect_noaa_weather_data.log'
SPARK_COMBINE_NOAA_WEATHER_DATA_LOG = './logs/spark_combine_noaa_weather_data.log'
SPARK_FIRE_AND_WEATHER_DATA_LOG = './logs/spark_combine_fire_and_weather_data.log'
SPARK_SELECT_FEATURES_LOG = './logs/spark_select_features.log'
SPARK_ADD_FEATURES_LOG = './logs/spark_select_features.log'