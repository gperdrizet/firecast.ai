from tensorflow import keras
import helper_functions.data_functions as data_functions

raw_data_file = '/mnt/SSD/data/training_data/1992-2015_california.parquet'
raw_data_box_file = '/mnt/SSD/data/training_data/1992-2002_california_box.parquet'
training_data_file = '/mnt/SSD/data/training_data/1992-2015_california_features_added.parquet'
lat_lon_bins_file = '/mnt/SSD/data/spatial_data/california_geospatial_bins.parquet'
states_shapefile = '/mnt/SSD/data/spatial_data/cb_2018_us_state_500k.shp'

quantile_tansformer_file = './data_transformations/quantile_transformer'
min_max_scaler_file = './data_transformations/min_max_scaler'

features = [
    'lat',
    'lon',
    'mean_air_2m',
    'mean_apcp',
    'mean_rhum_2m',
    'mean_dpt_2m',
    'mean_pres_sfc',
    'mean_uwnd_10m',
    'mean_vwnd_10m',
    'mean_cloud_cover',
#     'veg',
#     'crain',
    'ignition',
    'date'
]

features_to_scale = [
    'mean_air_2m',
    'mean_apcp',
    'mean_rhum_2m',
    'mean_dpt_2m',
    'mean_pres_sfc',
    'mean_uwnd_10m',
    'mean_vwnd_10m',
    'mean_cloud_cover',
#     'veg',
#     'crain'
]

months = {
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
}

# Distribution plotting variables

left  = 0.125  # the left side of the subplots of the figure
right = 0.65   # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.5   # the amount of width reserved for blank space between subplots
hspace = 0.7   # the amount of height reserved for white space between subplots

fig_rows = 2 #3
fig_cols = 4
plot_height = 7
plot_width = 20

plot_locations = [
    (0,0),(0,1),(0,2),(0,3),
    (1,0),(1,1),(1,2),(1,3),
#     (2,0),(2,1),(2,2),(2,3)
]

# metrics = [
#     keras.metrics.TruePositives(name='true_positives'),
#     keras.metrics.FalsePositives(name='false_positives'),
#     keras.metrics.TrueNegatives(name='true_negatives'),
#     keras.metrics.FalseNegatives(name='false_negatives'), 
#     keras.metrics.AUC,#(name='auc'),
#     data_functions.matthews_correlation,
#     data_functions.f1
# ]

shape_parameters = {
    'history_size': 5,         # size of past history time chunk
    'target_size': 1,          # number of timepoints to predict from each history time chunk
    'step': 1                  # number of timepoints to move the history time chunk
}