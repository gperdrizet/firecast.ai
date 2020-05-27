import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer


def scale_features(data: 'DataFrame', features_to_scale: list) -> 'DataFrame':

    qt = QuantileTransformer(random_state=0, output_distribution='normal')
    normalized_data = pd.DataFrame(qt.fit_transform(
        data[features_to_scale]), columns=features_to_scale)
    data[features_to_scale] = normalized_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(data[features_to_scale])
    data[features_to_scale] = scaled_features

    data[features_to_scale] = data[features_to_scale].astype('float32')

    return data
