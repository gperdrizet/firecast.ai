import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer


def scale_features(data: 'DataFrame', features_to_scale: list, quantile_transformer: 'QuantileTransformer', min_max_scaler: 'MinMaxScaler') -> 'DataFrame':
    '''Takes data and list of weather features, prefit quantile transformer
    and min max scaler. Returns dataframe with weather normalized weather
    features scaled to range of (-1, 1)'''

    # run box cox transform on features of intrest
    normalized_data = pd.DataFrame(quantile_transformer.transform(
        data[features_to_scale]), columns=features_to_scale)
    data[features_to_scale] = normalized_data

    # scale features of intrest
    scaled_features = min_max_scaler.transform(data[features_to_scale])
    data[features_to_scale] = scaled_features

    # reset dtypes of transformed features
    data[features_to_scale] = data[features_to_scale].astype('float32')

    # return resulting dataframe
    return data
