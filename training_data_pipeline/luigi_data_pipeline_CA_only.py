import luigi
import os
import shutil
# import requests
# import json
# import csv
# import numpy as np
# import pandas as pd
import CA_only.config as config

from CA_only.get_noaa_weather_data import get_weather_data
from CA_only.parse_noaa_weather_data import parse_weather_data
from CA_only.combine_noaa_weather_datafiles import combine_weather_data
from CA_only.get_usda_fire_data import get_fire_data


class GetWeatherData(luigi.Task):
    def requires(self):
        return None

    def output(self):
        finished_output_path = f'{config.WEATHER_DATA_BASE_PATH}{config.COMPLETE_RAW_DATAFILE_SUBDIR}'
        return luigi.LocalTarget(finished_output_path)

    def run(self):
        base_url = config.NOAA_BASE_URL
        data_years = config.DATA_YEARS
        data_types = config.DATA_TYPES.keys()
        temp_output_path = f'{config.WEATHER_DATA_BASE_PATH}{config.TEMP_RAW_DATAFILE_SUBDIR}'

        try:
            os.mkdir(temp_output_path)

        except:
            print('Temp directory exists')

        result = get_weather_data(
            base_url, data_years, data_types, temp_output_path)

        if result == True:
            print(f'Moving temp directory to: {self.output().path}')
            shutil.move(temp_output_path, self.output().path)


class ParseWeatherData(luigi.Task):
    def requires(self):
        return GetWeatherData()

    def output(self):
        finished_output_path = f'{config.WEATHER_DATA_BASE_PATH}{config.COMPLETE_PARSED_DATAFILE_SUBDIR}'
        return luigi.LocalTarget(finished_output_path)

    def run(self):
        data_years = config.DATA_YEARS
        temp_output_path = f'{config.WEATHER_DATA_BASE_PATH}{config.TEMP_PARSED_DATAFILE_SUBDIR}'

        try:
            os.mkdir(temp_output_path)

        except:
            print('Temp directory exists')

        result = parse_weather_data(data_years)

        if result == True:
            print(f'Moving temp directory to: {self.output().path}')
            shutil.move(temp_output_path, self.output().path)


class CombineWeatherData(luigi.Task):
    def requires(self):
        return ParseWeatherData()

    def output(self):
        output_file = f'{config.WEATHER_DATA_BASE_PATH}all.{str(config.START_YEAR)}-{str(config.END_YEAR)}.california_only.parquet'
        return luigi.LocalTarget(output_file)

    def run(self):
        data_years = config.DATA_YEARS
        output_data = combine_weather_data(data_years)

        output_data.to_parquet(self.output().path)


class GetWildfireData(luigi.Task):
    def requires(self):
        return CombineWeatherData()

    def output(self):
        output_file = f'{config.FIRE_DATA_BASE_PATH}{config.FIRE_DATA_FILE}'
        return luigi.LocalTarget(output_file)

    def run(self):
        url = config.USDA_URL
        path = config.FIRE_DATA_BASE_PATH
        file_name = config.FIRE_SQL_FILE
        output_data = get_fire_data(url, path, file_name)

        # output_data.to_parquet(self.output().path)


if __name__ == '__main__':
    luigi.run()
