import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer


# def scale_features(data: 'DataFrame', features_to_scale: list, features_to_transform: list, standard_scaler: 'StandardScaler', quantile_transformer: 'QuantileTransformer') -> 'DataFrame':
def scale_features(data: 'DataFrame', features_to_scale: list, min_max_scaler: 'MinMaxScaler') -> 'DataFrame':
    '''Takes data and list of weather features, prefit quantile transformer
    and standard scaler. Returns dataframe with weather normalized and
    weather transformed weather features'''

    print(type(min_max_scaler))
    # print(type(quantile_transformer))

    data['raw_lat'] = data['lat']
    data['raw_lon'] = data['lon']

    print(data.info())

    # scale features of intrest
    scaled_data = pd.DataFrame(min_max_scaler.transform(
        data[features_to_scale]), columns=features_to_scale)
    data[features_to_scale] = scaled_data

    # run quantile transform on features of intrest
    # transformed_data = pd.DataFrame(quantile_transformer.transform(
    #     data[features_to_transform]), columns=features_to_transform)
    # data[features_to_transform] = transformed_data

    # reset dtypes of transformed features
    data[features_to_scale] = data[features_to_scale].astype('float32')

    # return resulting dataframe
    return data
