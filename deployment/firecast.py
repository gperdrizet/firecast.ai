import luigi
import requests
import json
import csv
import numpy as np
import pandas as pd
import config
import keys

from pickle import load
from datetime import date, datetime, timedelta

from data_functions.get_data import get_current_fires
from data_functions.get_data import get_weather_forecast
from data_functions.get_data import get_past_weather
from data_functions.parse_weather_data import parse_data
from data_functions.scale_weather_features import scale_features
from data_functions.onehot_encode_month import onehot_month
from data_functions.format_for_lstm import format_data
#from prediction_functions.predict_ignition_risk import predict
from prediction_functions.format_predictions_for_api import format_for_api

TODAY = str(date.today())
YESTERDAY = str(date.today() - timedelta(1))
print(f'Running prediction pipeline for: {TODAY}')
print(f'Will get past data for: {YESTERDAY}')


class GetActiveFires(luigi.Task):
    '''Gets active fires from https://data-nifc.opendata.arcgis.com'''

    def requires(self):
        # First task in pipeline, requires nothing
        return None

    def output(self):
        # Output is JSON fire data file named for today's date
        output_file = f'{config.RAW_DATA_DIR}fires/{TODAY}.json'
        return luigi.LocalTarget(output_file)

    def run(self):

        # run functions to download fire data
        fire_data = get_current_fires(TODAY)

        # save result as JSON
        with self.output().open('w') as output_file:
            json.dump(fire_data, output_file)


class GetWeatherForecast(luigi.Task):
    '''Gets weather forecast data from openweathermap.org API'''

    def requires(self):
        # First task in pipeline, requires nothing
        return None

    def output(self):
        # Output is JSON weather data file named for today's date
        output_file = f'{config.RAW_DATA_DIR}future_weather/{TODAY}.json'
        return luigi.LocalTarget(output_file)

    def run(self):
        # get API key as string
        key = keys.OPENWEATHER_API_KEY

        # file name string for files contating the latitude, longitude corrdinates
        # we need to get weather data for
        lat_lon_bin_file = config.LAT_LON_BINS_FILE

        # read lat, lon bins into list
        with open(lat_lon_bin_file, 'r') as lat_lon_bins_file:
            rows = csv.reader(lat_lon_bins_file)
            lat_lon_bins = list(rows)

        # run functions to download weather data from API
        weather_data = get_weather_forecast(
            key, lat_lon_bins, TODAY)

        # save result as JSON
        with self.output().open('w') as output_file:
            json.dump(weather_data, output_file)


class GetPastWeather(luigi.Task):
    '''Gets one day of past weather data from openweathermap.org API'''

    def requires(self):
        # First task in pipeline, requires nothing
        return None

    def output(self):
        # Output is JSON weather data file named for today's date
        output_file = f'{config.RAW_DATA_DIR}past_weather/{YESTERDAY}.json'
        return luigi.LocalTarget(output_file)

    def run(self):
        # get API key as string
        key = keys.OPENWEATHER_API_KEY

        # file name string for files contating the latitude, longitude corrdinates
        # we need to get weather data for
        lat_lon_bin_file = config.LAT_LON_BINS_FILE

        # read lat, lon bins into list
        with open(lat_lon_bin_file, 'r') as lat_lon_bins_file:
            rows = csv.reader(lat_lon_bins_file)
            lat_lon_bins = list(rows)

        # run functions to download weather data from API
        weather_data = get_past_weather(key, lat_lon_bins, YESTERDAY)

        # save result as JSON
        with self.output().open('w') as output_file:
            json.dump(weather_data, output_file)


class ParseWeatherData(luigi.Task):
    '''Takes raw weather data, retireives and formats weather
    variables of intrest'''

    def requires(self):
        # requires successful download of today's past and future weather data
        return GetWeatherForecast(), GetPastWeather(), GetActiveFires()

    def output(self):
        # output is csv file named for today's date in intermediate data
        # processing directory
        output_file = f'{config.INTERMIDIATE_DATA_DIR}weather/{TODAY}.parquet'
        return luigi.LocalTarget(output_file)

    def run(self):

        # features to extract from weather data
        column_names = config.WEATHER_DATA_COLUMN_NAMES
        # run function to parse, extract and format weather data
        output_data = parse_data(TODAY, 5, 7, column_names)

        print(f'Will save parsed weather data to: {self.output().path}')
        print(output_data.head())
        # write output
        output_data.to_parquet(self.output().path)


class ScaleWeatherFeatures(luigi.Task):
    '''Nomalize weather data with box cox transformation
    then scale to range (-1, 1), matches tanh activation function
    in LSTM layers'''

    def requires(self):
        # requires parsed weather data
        return ParseWeatherData()

    def output(self):
        # writes result to intermidate data processing dir, names after today's
        # date with extension
        output_file = f"{config.INTERMIDIATE_DATA_DIR}weather/{TODAY}_scaled.parquet"
        return luigi.LocalTarget(output_file)

    def run(self):
        # load parsed weather data
        input_data = pd.read_parquet(self.input().path)

        # load box cox transformer and min max scalers which were used
        # to scale the training data - this way our live prediction data
        # is transformed in exactly the same way
        quantile_tansformer_file = f'{config.DATA_TRANSFORMATION_DIR}quantile_transformer'
        min_max_scaler_file = f'{config.DATA_TRANSFORMATION_DIR}min_max_scaler'
        quantile_tansformer = load(open(quantile_tansformer_file, 'rb'))
        min_max_scaler = load(open(min_max_scaler_file, 'rb'))

        # run function to transform and scale features of intrest
        features_to_scale = config.WEATHER_FEATURES_TO_SCALE
        output_data = scale_features(
            input_data, features_to_scale, quantile_tansformer, min_max_scaler)

        # write result
        output_data.to_parquet(self.output().path)


class OneHotEncodeMonth(luigi.Task):
    '''One hot encodes month'''

    def requires(self):
        # Takes scaled weather data
        return ScaleWeatherFeatures()

    def output(self):
        # writes result to intermidate data processing dir, names after today's
        # date with descriptive extension
        output_file = f"{config.INTERMIDIATE_DATA_DIR}weather/{TODAY}_scaled_onehot_month.parquet"
        return luigi.LocalTarget(output_file)

    def run(self):
        # read scaled weather data
        input_data = pd.read_parquet(self.input().path)

        # run fucntion to one hot encode month
        output_data = onehot_month(input_data)

        # write result to disk
        output_data.to_parquet(self.output().path)


class FormatForLSTM(luigi.Task):
    '''Takes processed weather data and formats for imput to LSTM neural net for
    fire risk prediction.'''

    def requires(self):
        # takes complete transformed,scaled and encoded weather dataset
        return OneHotEncodeMonth()

    def output(self):
        # saves result to pickled numpy array of arrays
        output_file = f"{config.PROCESSED_DATA_DIR}weather/{TODAY}_input_weather_data.npy"
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        # read input data
        input_data = pd.read_parquet(self.input().path)

        # load target shape parameters for numpy array
        input_shape_parameters = config.LSTM_INPUT_SHAPE_PARAMETERS

        # run function to do shape transforms
        output_data = format_data(input_data, input_shape_parameters)

        # convert to numpy
        output_data = np.asarray(output_data)

        # save as npy fule
        with self.output().open('wb') as output_file:
            np.save(output_file, output_data)


class Predict(luigi.Task):
    '''Takes formatted weather data, model weights and hyperparameters,
    predicts fire igintion risk'''

    def requires(self):
        # requires formatted weather dataset
        return FormatForLSTM()

    def output(self):
        # writes predictions to npy file named after today's date
        output_file = f"{config.IGNITION_RISK_PREDICTIONS_DIR}{TODAY}_predictions.npy"
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        # load weather data
        with self.input().open('r') as input_file:
            input_data = np.load(input_file)

        data = json.dumps(
            {"signature_name": "serving_default", "instances": input_data.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(
            'http://localhost:8501/parallel_LSTM/versions/1:predict', data=data, headers=headers)
        print(f'Post response code: {json_response.status_code}')
        predictions = json.loads(json_response.text)['predictions']
        #predictions = json.loads(json_response.text)
        print(predictions)
        # save predictions
        with self.output().open('wb') as output_file:
            np.save(output_file, predictions)


class FormatPredictionsForAPI(luigi.Task):
    '''Takes raw prediction results from LSTM neural network and formats
    them to be served by flask API'''

    def requires(self):
        # needs today's predictions
        return Predict()

    def output(self):
        # writes formatted predictions to parquet file named for today's date
        # with descriptive extension
        output_file = f"{config.IGNITION_RISK_PREDICTIONS_DIR}{TODAY}_formatted_predictions.parquet"
        return luigi.LocalTarget(output_file)

    def run(self):
        # load unformatted predictions
        with self.input().open('r') as input_file:
            input_data = np.load(input_file)

        # load latitude and longitude bins that predictions
        # correspond to - read into list
        lat_lon_bin_file = config.LAT_LON_BINS_FILE

        with open(lat_lon_bin_file, 'r') as lat_lon_bins_file:
            rows = csv.reader(lat_lon_bins_file)
            lat_lon_bins = list(rows)

        # run fucntion to format predictions
        output_data = format_for_api(input_data, lat_lon_bins)

        print(output_data.head())
        print(output_data.info())

        # write result to disk
        output_data.to_parquet(self.output().path)


if __name__ == '__main__':
    luigi.run()
