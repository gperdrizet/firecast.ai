import luigi
import requests
import json
import csv
import pandas as pd
import config
import keys

from data.get_weather_data import get_data
from data.parse_weather_data import parse_data
from data.scale_weather_features import scale_features
from data.onehot_encode_month import onehot_month


class GetWeatherData(luigi.Task):
    def requires(self):
        return None

    def output(self):
        output_file = f"{config.RAW_WEATHER_DATA_DIR}test.json"
        return luigi.LocalTarget(output_file)

    def run(self):
        key = keys.OPENWEATHER_API_KEY
        lat_lon_bin_file = config.LAT_LON_BINS_FILE

        with open(lat_lon_bin_file, 'r') as lat_lon_bins_file:
            rows = csv.reader(lat_lon_bins_file)
            lat_lon_bins = list(rows)

        weather_data = get_data(key, lat_lon_bins)

        with self.output().open('w') as output_file:
            json.dump(weather_data, output_file)


class ParseWeatherData(luigi.Task):
    def requires(self):
        return GetWeatherData()

    def output(self):
        output_file = f"{config.INTERMIDIATE_WEATHER_DATA_DIR}test.csv"
        return luigi.LocalTarget(output_file)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = json.load(input_file)

        output_data = parse_data(
            input_data, config.WEATHER_DATA_COLUMN_NAMES)

        with self.output().open('w') as output_file:
            output_data.to_csv(output_file, index=False)


class ScaleWeatherFeatures(luigi.Task):
    def requires(self):
        return ParseWeatherData()

    def output(self):
        output_file = f"{config.INTERMIDIATE_WEATHER_DATA_DIR}test_scaled.csv"
        return luigi.LocalTarget(output_file)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = pd.read_csv(input_file)

        features_to_scale = config.WEATHER_FEATURES_TO_SCALE
        output_data = scale_features(input_data, features_to_scale)

        with self.output().open('w') as output_file:
            output_data.to_csv(output_file, index=False)


class OneHotEncodeMonth(luigi.Task):
    def requires(self):
        return ScaleWeatherFeatures()

    def output(self):
        output_file = f"{config.INTERMIDIATE_WEATHER_DATA_DIR}test_scaled_onehot_month.csv"
        return luigi.LocalTarget(output_file)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = pd.read_csv(input_file)

        output_data = onehot_month(input_data)

        with self.output().open('w') as output_file:
            output_data.to_csv(output_file, index=False)


if __name__ == '__main__':
    luigi.run()
