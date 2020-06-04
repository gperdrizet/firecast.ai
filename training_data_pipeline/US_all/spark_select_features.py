'''Select weather variables to become features in the training data.'''

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
    filename=config.SPARK_SELECT_FEATURES_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

# connect to spark cluster and set up spark context
jobname = os.path.basename(__file__)
sc = pyspark.SparkContext(master=config.SPARK_MASTER, appName=jobname)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

logging.info(sqlContext)

# construct file names
input_data_file = (config.TRAINING_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + '_us_training_data_raw.parquet')
output_data_file = (config.TRAINING_DATA_BASE_PATH + str(config.START_YEAR) + "-" + str(config.END_YEAR) + '_us_training_data_processed.parquet')

# read data
data = spark.read.parquet(input_data_file)

# replace any NA values with 0
data = data.na.fill(0)

# write the desired features to disk
output_data = data.select(*config.FEATURES)
output_data.write.format("parquet").mode("overwrite").save(output_data_file)