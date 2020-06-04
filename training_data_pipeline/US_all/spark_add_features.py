'''Smooth data with a rolling window and add min, max and mean
features'''

import findspark
findspark.init()

import pandas as pd
import pyspark
import logging
import os.path
import config
import numpy as np
import datetime as dt

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as func

# set up logging
logging.basicConfig(
    filename=config.SPARK_ADD_FEATURES_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

# connect to spark cluster and set up spark context
jobname = os.path.basename(__file__)
sc = pyspark.SparkContext(master=config.SPARK_MASTER, appName=jobname)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

logging.info(sqlContext)

# Location of raw data
input_data_file = (config.TRAINING_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + '_us_training_data_processed.parquet')
output_data_file = (config.TRAINING_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + '_us_training_data_features_added.parquet')

# read data
data = spark.read.parquet(input_data_file)
#data = data.sort(func.desc("time")) # DEFINITLY sort before repartition... Do we even need to repartition?
#data = data.repartition(num_partitions, 'lat', 'lon')
#print("Number of partitions: "+str(data.rdd.getNumPartitions()))

# Add mean features
input_features = (
#     'time', # Note: daily avg. data has no time column, just day, month, year
    'air_2m',
    'apcp',
    'rhum_2m',
    'dpt_2m',
    'pres_sfc',
    'uwnd_10m', 
    'vwnd_10m',
    'veg',
    'vis',
#     'lat',
#     'lon',
    'ignition'
)

def seconds_from_days(days):
    return days * 86400

window = Window.partitionBy("lat", "lon").orderBy(func.col("time").cast('long')).rangeBetween(0, seconds_from_days(3))

# Names for new columns
new_features = (
    'mean_air_2m',
    'mean_apcp',
    'mean_rhum_2m',
    'mean_dpt_2m',
    'mean_pres_sfc',
    'mean_uwnd_10m', 
    'mean_vwnd_10m',
    'mean_veg',
    'mean_vis',
    'mean_ignition'
)

for i in range(len(input_features)):
    input_feature = input_features[i]
    new_feature = new_features[i]
    print("Calculating 3 day mean for "+input_feature)
    data = data.withColumn(new_feature, func.avg(input_feature).over(window))
    
    # Names for new columns
new_features = (
    'min_air_2m',
    'min_apcp',
    'min_rhum_2m',
    'min_dpt_2m',
    'min_pres_sfc',
    'min_uwnd_10m', 
    'min_vwnd_10m',
    'min_veg',
    'min_vis'
)

for i in range(len(input_features) - 1):
    input_feature = input_features[i]
    new_feature = new_features[i]
    print("Calculating 3 day min for "+input_feature)
    data = data.withColumn(new_feature, func.min(input_feature).over(window))
    
    # Names for new columns
new_features = (
    'max_air_2m',
    'max_apcp',
    'max_rhum_2m',
    'max_dpt_2m',
    'max_pres_sfc',
    'max_uwnd_10m', 
    'max_vwnd_10m',
    'max_veg',
    'max_vis'
)

for i in range(len(input_features) - 1):
    input_feature = input_features[i]
    new_feature = new_features[i]
    print("Calculating 3 day max for "+input_feature)
    data = data.withColumn(new_feature, func.max(input_feature).over(window))
    
data.write.format("parquet").mode("overwrite").save(output_data_file)