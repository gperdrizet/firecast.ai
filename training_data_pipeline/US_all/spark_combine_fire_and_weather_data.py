'''Combine weather and fire data using left outer join on location and data'''
import findspark
findspark.init()

import pandas as pd
import pyspark
import numpy as np
import logging
import os.path
import config

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import expr
from pyspark.sql.functions import lit
import pyspark.sql.functions as func

# set up logging
logging.basicConfig(
    filename=config.SPARK_FIRE_AND_WEATHER_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

# connect to spark cluster and set up spark context
jobname = os.path.basename(__file__)
sc = pyspark.SparkContext(master=config.SPARK_MASTER, appName=jobname)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# construct data file paths
weather_data_file = (config.WEATHER_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + "_us_only_all.parquet")
fire_data_file = (config.FIRE_DATA_BASE_PATH + 'regridded_us_fires.parquet')
training_data_file = '/mnt/SSD/data/training_data/1992-2015_us_training_data_raw.parquet'
training_data_file = (config.TRAINING_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + '_us_training_data_raw.parquet')
years = config.END_YEAR - config.START_YEAR + 1

weather = spark.read.parquet(weather_data_file)
weather = weather.drop('__index_level_0__')
weather = weather.withColumn("date", expr("to_date(time)"))
#weather = weather.repartition((years * 365), 'date')
#print("Number of partitions: "+str(weather.rdd.getNumPartitions()))
#print("Number of unique dates: "+str(weather.select(func.countDistinct("date")))) # throws weird error
#weather.show(n=1, truncate=True, vertical=True)

fires = spark.read.parquet(fire_data_file)
fires = fires.drop('__index_level_0__')
fires = fires.withColumn("date", expr("to_date(date)"))
fires = fires.withColumn("ignition", lit(1))
fires = fires.withColumnRenamed("time", "discovery_time")
#fires = fires.repartition(8400, 'date')
#print("Number of partitions: "+str(fires.rdd.getNumPartitions()))
#print("Number of unique dates: "+str(fires.select(func.countDistinct("date")))) # throws weird error
#fires.show(n=1, truncate=True, vertical=True)

training_data = weather.join(fires, ['lat', 'lon', 'date'], 'left_outer')#how='left_outer')
# Note: first time training_data is actualy evaluated takes ~12 min.

#print("Number of partitions: "+str(training_data.rdd.getNumPartitions()))

#training_data.write.format("parquet").mode("overwrite").save(training_data_file)
training_data.write.format("parquet").mode("overwrite").save(training_data_file)