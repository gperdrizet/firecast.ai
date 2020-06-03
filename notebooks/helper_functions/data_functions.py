import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.python.keras import backend as K

def multivariate_data(
    dataset,
    history_size,
    target_size, 
    step
):
    
    # sort by date, then drop date column
    dataset = dataset.sort_values('date')
    dataset.drop('date', axis=1, inplace=True)

    # get index of ignition column
    ignition_index = dataset.columns.get_loc('ignition')

    # convert to numpy array
    dataset = np.array(dataset.values)

    # split ignition label off
    target = dataset[:, ignition_index]
    dataset = np.delete(dataset, ignition_index, 1)
    
    data = []
    labels = []

    start_index = history_size
    end_index = len(dataset) - target_size
    
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        labels.append(target[i + target_size])

    return np.array(data), np.array(labels)


def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1_val = 2 * ((precision*10) * recall) / ((precision*10) + recall + K.epsilon())
    
    return f1_val


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())