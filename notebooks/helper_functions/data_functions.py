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
    '''calculates harmonic mean of precision and recall
    Note: this function was removed from keras'''
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1_val = 2 * ((precision*10) * recall) / ((precision*10) + recall + K.epsilon())
    
    return f1_val


def matthews_correlation(y_true, y_pred):
    '''Calculates MCC from predicted and true lables'''
    
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

def format_features_labels_for_LSTM(
    dataframe,            # incomming_data
    history_size,         # size of past history time chunk
    target_size,          # number of timepoints to predict from each history time chunk
    step                  # number of timepoints to move the history time chunk as we 
                          # slide over the data
):
    
    '''takes dataframe and information about desired output shape (history window size, 
    target window size, timesteps to slide history window) and returns two formatted
    lists: one for features and one for labels'''

    data = []
    labels = []
    
    # break dataframe into spatial bins by lat lon
    spatial_bins = dataframe.groupby(['lat', 'lon'])
    
    # loop on spatial bins and split each into test and training 
    # sets 
    for bin_name, spatial_bin in spatial_bins:
        
        # sort bin by date, then drop date column
        spatial_bin = spatial_bin.sort_values('date')
        spatial_bin.drop('date', axis=1, inplace=True)
        
        # get index of ignition column
        ignition_index = spatial_bin.columns.get_loc('ignition')
        
        # convert to numpy array
        spatial_bin = np.array(spatial_bin.values)
        
        # split ignition label off
        target = spatial_bin[:, ignition_index]
        spatial_bin = np.delete(spatial_bin, ignition_index, 1)
        
        bin_data = []
        bin_labels = []
    
        # set start and end - note: some trimming is 
        # necessary here so that the target (in the future)
        # does not slide off the end of the array as we
        # move the history window forward
        start_index = history_size
        end_index = len(spatial_bin) - target_size

        # loop over the history window in steps of step size
        # grab a time block of data with length history_size
        # and the corresponding labels
        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            bin_data.append(spatial_bin[indices])
            bin_labels.append(target[i + target_size])

        # add to master
        data.append(np.array(bin_data))
        labels.append(np.array(bin_labels))

    return data, labels


def trim_and_reshape_for_LSTM(x, y):
    '''Trim the number of samples in each bin so that they are all the same.
    at the same time, reshape y so that it's first axis is the number of samples
    and the second is number of spatial bins'''
    
    sample_sizes = []

    for sample in y:
        sample_sizes.append(len(sample))

    smallest_sample = min(sample_sizes)

    y_reshaped = []

    for i in range(smallest_sample):
        new_y = []
        for j in range(len(y)):
            try:
                new_y.append(y[j][i])
            except:
                print("Index out of range")

        y_reshaped.append(np.array(new_y))

    trimmed_x = []    

    for sample in x:
        trimmed_sample = sample[-smallest_sample:,:]
        trimmed_x.append(trimmed_sample)
        
        
    return trimmed_x, y_reshaped

def weighted_bce(class_0_weight, class_1_weight):
    '''Implements weighted binary cross-entropy loss for multiclass classification'''
    def loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * ((y_true * class_1_weight) + class_0_weight))
    
    return loss