import luigi
import requests
import json
import csv
import numpy as np
import pandas as pd
import config
import keys

from data.get_weather_data import get_data
from data.parse_weather_data import parse_data
from data.scale_weather_features import scale_features
from data.onehot_encode_month import onehot_month
from data.format_for_lstm import format_data
from prediction.predict_ignition_risk import predict
from prediction.format_predictions_for_api import format_for_api


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

        quantile_tansformer = config.QUANTILE_TRANSFORMER_FILE
        min_max_scaler = config.MIN_MAX_SCALER_FILE
        features_to_scale = config.WEATHER_FEATURES_TO_SCALE
        output_data = scale_features(
            input_data, features_to_scale, quantile_transformer, min_max_scaler)

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


class FormatForLSTM(luigi.Task):
    def requires(self):
        return OneHotEncodeMonth()

    def output(self):
        output_file = f"{config.PROCESSED_WEATHER_DATA_DIR}input_weather_data.npy"
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = pd.read_csv(input_file)

        input_shape_parameters = config.LSTM_INPUT_SHAPE_PARAMETERS
        output_data = format_data(input_data, input_shape_parameters)
        output_data = np.asarray(output_data)

        with self.output().open('wb') as output_file:
            np.save(output_file, output_data)


class Predict(luigi.Task):
    def requires(self):
        return FormatForLSTM()

    def output(self):
        output_file = f"{config.IGNITION_RISK_PREDICTIONS_DIR}predictions.npy"
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = np.load(input_file)

        weights_file = config.TRAINED_MODEL_WEIGHTS_FILE
        hyperparameters = config.LSTM_HYPERPARAMETERS
        output_data = predict(input_data, hyperparameters, weights_file)

        print(output_data)

        with self.output().open('wb') as output_file:
            np.save(output_file, output_data)


class FormatPredictionsForAPI(luigi.Task):
    def requires(self):
        return Predict()

    def output(self):
        output_file = f"{config.IGNITION_RISK_PREDICTIONS_DIR}formatted_predictions.csv"
        return luigi.LocalTarget(output_file)

    def run(self):
        with self.input().open('r') as input_file:
            input_data = np.load(input_file)

        lat_lon_bin_file = config.LAT_LON_BINS_FILE

        with open(lat_lon_bin_file, 'r') as lat_lon_bins_file:
            rows = csv.reader(lat_lon_bins_file)
            lat_lon_bins = list(rows)

        output_data = format_for_api(input_data, lat_lon_bins)

        with self.output().open('w') as output_file:
            output_data.to_csv(output_file, index=False)


if __name__ == '__main__':
    luigi.run()
