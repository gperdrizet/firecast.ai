import numpy as np
import pandas as pd
from pickle import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer


def scale_features(data: 'DataFrame', features_to_scale: list, quantile_transformer: str, min_max_scaler: str) -> 'DataFrame':

    quantile_transformer = load(open(quantile_transformer, 'rb'))
    normalized_data = pd.DataFrame(quantile_transformer.transform(
        data[features_to_scale]), columns=features_to_scale)
    data[features_to_scale] = normalized_data

    min_max_scaler = load(open(min_max_scaler, 'rb'))
    scaled_features = min_max_scaler.transform(data[features_to_scale])
    data[features_to_scale] = scaled_features

    data[features_to_scale] = data[features_to_scale].astype('float32')

    return data
