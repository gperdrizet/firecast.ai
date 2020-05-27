import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def onehot_month(data: 'DataFrame') -> 'DataFrame':
    month_names = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December'
    ]

    onehot_encoder = OneHotEncoder(
        categories=[np.arange(1, 13, 1)],
        sparse=False,
        dtype=np.float32
    )

    # get months from each row of data, reshape from wide to long
    month = np.array(pd.DatetimeIndex(data['date']).month).reshape(-1, 1)

    # onehot encode
    onehot_month = onehot_encoder.fit_transform(month)
    # print(onehot_month)

    # convert to pandas dataframe with named columns
    onehot_month_df = pd.DataFrame(onehot_month, columns=month_names)

    # reset indexes
    onehot_month_df.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # join months back to data along rows
    data = pd.concat([data, onehot_month_df], axis=1)

    return data
