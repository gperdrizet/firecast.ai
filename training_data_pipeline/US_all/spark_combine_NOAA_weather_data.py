'''Combines weather data for each year into a single file'''

import findspark
findspark.init()

import pandas as pd
import pyspark
import logging
import os.path
import config
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession

# set up logging
logging.basicConfig(
    filename=config.SPARK_COMBINE_NOAA_WEATHER_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

# connect to spark cluster and set up spark context
jobname = os.path.basename(__file__)
sc = pyspark.SparkContext(master=config.SPARK_MASTER, appName=jobname)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

logging.info(sqlContext)

# unpack data years into list
data_years = [*config.DATA_YEARS]

# read the first year of data into a master dataframe so we have something to join subsiquent years with
input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR + str(data_years.pop(0)) + "_us_only_all.parquet")
master_df = spark.read.parquet(input_file)

# loop on years, joining each to the master dataframe
for year in data_years:
    input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR + str(year) + "_us_only_all.parquet")
    df = spark.read.parquet(input_file)
    master_df = master_df.union(df)
    
# wrrite output 
output_file = (config.WEATHER_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + "_us_only_all.parquet")
master_df.write.format("parquet").mode("overwrite").save(output_file)