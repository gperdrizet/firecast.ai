import random
from statistics import mean
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import pandas as pd
import geopandas as gpd
from tensorflow.python.keras import backend as K
from scipy.interpolate import griddata

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix

def get_california_polygon(shapefile):
    gdf = gpd.read_file(shapefile)
    california = gdf[gdf['NAME'] == 'California']
    return(california)

def regularize_grid(data, data_type, GRID_SPACING):
    
    # data coordinates and values
    x = data['lon']
    y = data['lat']
    z = data[data_type]

    # target grid to interpolate to
    xi = np.arange(data['lon'].min(), data['lon'].max(), GRID_SPACING)
    yi = np.arange(data['lat'].min(), data['lat'].max(), GRID_SPACING)
    xi,yi = np.meshgrid(xi,yi)

    # interpolate
    zi = griddata((x,y),z,(xi,yi),method='linear')
    
    return xi, yi, zi

def k_random_sample(data, k):
    # Takes a data frame and an number of observations
    # returns dataframe containing k from n pseudorandom
    # observations with out replacement
    
    n = len(data)
    indices = random.sample(range(0, n), k)
    return data.iloc[indices]

def stratified_sample(data, n):
    # takes a datafram and a sample size n, returns
    # n observations with positive and negative class
    # frequency matched to orignal data
    
    # split positive and negative datsets up
    ignitions = data[data['ignition'] == 1]
    no_ignitions = data[data['ignition'] == 0]
    
    # Calculate ignition & no ignition sample sizes
    ignition_fraction = len(ignitions) / len(data)
    ignition_sample_size = int((n * ignition_fraction))
    no_ignition_sample_size = int((n * (1 - ignition_fraction)))
    
    # sample data
    no_ignitions_sample = k_random_sample(no_ignitions, no_ignition_sample_size)
    ignitions_sample = k_random_sample(ignitions, ignition_sample_size)

    # combine
    sampled_data = no_ignitions_sample.append(ignitions_sample)
    
    return sampled_data

def cross_validate_classifier(classifier, X_train, y_train, folds, scoring_func):
    # Takes a classifier, x and y training data, a number of folds for
    # cross validation and a scoring function. Runs cross validation and returns
    # array of scores from each fold
    
    cv = StratifiedKFold(n_splits=folds)
    cross_val_scores = cross_val_score(classifier, X_train, y_train, scoring=scoring_func, cv=cv)
    
    return cross_val_scores

def fit_model(classifier, X, y):
    # Takes classifier, data and labels, returns
    # fit model
    classifier.fit(X, y)
    return classifier

def display_confusion_matrix(classifier, class_names, x_test, y_test):
    # Takes a fit classifier, the class names as a list and
    # test data. Prints raw confusion matrix and plots
    # normalized confusion matrix

    raw_cm = confusion_matrix(y_test, classifier.predict(x_test))
    print("Raw count confusion matrix")
    print(raw_cm)
    
    normalized_cm = plot_confusion_matrix(
        classifier,
        x_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize='true'
    )

    normalized_cm.ax_.set_title("Normalized confusion matrix")

    plt.show()
    
def calc_false_neg_pos_rate(model, x_test, y_test):
    # Takes a fit model and test data, returns 
    # false positive and negative rates
    
    cm = confusion_matrix(y_test, model.predict(x_test))

    TN = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]

    false_neg_rate = FN / (FN + TN)
    false_pos_rate = FP / (FP + TN)
    
    return false_neg_rate, false_pos_rate

def scale_weather_variables(weather_variables, X_train, X_test):
    # Uses StandardScaler to convert data to Z-score. Takes list of
    # weather variable names to scale and train/test data. Calculates
    # transformation from test data, applies same transform to
    # train and test data, returns scaled data.

    scaler = StandardScaler()
    scaler.fit(X_train[weather_variables])

    pd.options.mode.chained_assignment = None  # default='warn'
    
    X_train[weather_variables] = scaler.transform(X_train[weather_variables])
    X_test[weather_variables] = scaler.transform(X_test[weather_variables])
    
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    
    return X_train, X_test

def make_train_test_sample(dataset, sample_size, train_test_split_ratio, rand_seed):
    # Takes data in dataframe, sample size, train test split
    # ratio and random seed. Samples n datapoints from dataset and then
    # runs stratified train test split on sample.
    # return stratified train test split data
    
    column_names = dataset.columns
    
    sampled_data = stratified_sample(dataset, sample_size)

    y = sampled_data['ignition']
    X = sampled_data.drop('ignition', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=train_test_split_ratio, 
        random_state=rand_seed, 
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def FP_rate_scorer(y_test, y_pred):
    # Calulates false positive rate from test data and
    # predictions. For use with make_scorer
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    false_pos_rate = FP / (FP + TN)
    
    return false_pos_rate

def FN_rate_scorer(y_test, y_pred):
    # Calulates false positive rate from test data and
    # predictions. For use with make_scorer
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    false_neg_rate = FN / (FN + TN)
    
    return false_neg_rate

def tune_class_weight(
    class_weights,
    max_jobs,
    rand_seed,
    data,
    repitions,
    sample_size,
    train_test_split_ratio,
    cv_folds,
    scoring_func_name,
    scoring_func
):
    # Takes class weight list, number of parallel jobs
    # A random seed and training data. Itterates over
    # class weights. Stores and returns F1 score,
    # false positive and false negative rates for each
    # class weight in a data frame.
    
    model_scores_columns = [
        'Class weight',
        'CV {} score'.format(scoring_func_name),
        'CV False positive rate',
        'CV False negative rate'
    ]
    
    model_scores = pd.DataFrame(columns=model_scores_columns)

    for class_weight in class_weights:
        
        for i in range(repitions):
            
            # draw new stratifited sample and run train test split
            X_train, X_test, y_train, y_test = make_train_test_sample(
                data, 
                sample_size, 
                train_test_split_ratio, 
                rand_seed
            )

            classifier = XGBClassifier(
                n_jobs = max_jobs,
                scale_pos_weight = class_weight,
                random_state = rand_seed
            )

            classifier.fit(X_train, y_train)

            cross_val_scores = cross_validate_classifier(
                classifier, 
                X_train, 
                y_train, 
                cv_folds, 
                scoring_func
            )

            score = mean(cross_val_scores)

            cross_val_FP_rates = cross_validate_classifier(
                classifier, 
                X_train, 
                y_train, 
                cv_folds, 
                make_scorer(FP_rate_scorer)
            ) 

            cross_val_FP_rate = mean(cross_val_FP_rates)

            cross_val_FN_rates = cross_validate_classifier(
                classifier, 
                X_train, 
                y_train, 
                cv_folds, 
                make_scorer(FN_rate_scorer)
            ) 

            cross_val_FN_rate = mean(cross_val_FN_rates)

            model_scores = model_scores.append(pd.Series([
                class_weight,
                np.round(score,3), 
                np.round(cross_val_FP_rate,3), 
                np.round(cross_val_FN_rate,3)
            ], index=model_scores.columns), ignore_index=True)

    return model_scores

def tune_hyperparameters(
    classifier,
    param_dist, 
    X_train, 
    y_train, 
    num_jobs, 
    search_iterations, 
    scoring_func
):
    # Tunes arbitraty hyperparamter(s) takes a classifier
    # a parameter distribution as a dictionary, training data
    # a number of parallel jobs, a number of search iterations
    # and a scoring function to use for cross validation
    # returns winning model and cross validation results as
    # data frame

    random_search = RandomizedSearchCV(
        classifier, 
        param_distributions=param_dist,
        scoring=scoring_func,
        n_iter=search_iterations,
        n_jobs=num_jobs
    )

    best_model = random_search.fit(X_train, y_train)
    
    return best_model, random_search

def regularize_hyperparameter_grid(x, y, z, resolution):
    # Takes three coordinate grids and a resolution
    # interpolates and resamples so frequencys match
    # returns regularized data. For use in
    # constructing heatmaps

    # target grid to interpolate to
    xi = np.arange(min(x), max(x), ((max(x) - min(x)) / resolution))
    yi = np.arange(min(y), max(y), ((max(y) - min(y)) / resolution))
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return xi, yi, zi
    
def plot_relative_feature_importance(model, data, x_test, x_tick_size):
    # Takes a fit model, orignal data and test data, plots relative feature
    # importance
    
    column_names = x_test.columns
    
    if 'weather_bin_time' in column_names:
        x_test = x_test.drop('weather_bin_time', axis=1)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(list(x_test))

    plt.figure(figsize=(20,10))
    plt.rc('axes', titlesize=30)     # fontsize of the axes title
    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=x_tick_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
    plt.title("Feature importance")
    plt.bar(range(x_test.shape[1]), importances[indices],
           color="darkblue", align="center")
    plt.xticks(np.arange(len(indices)), feature_names[indices], rotation='vertical')
    plt.xlim([-1, x_test.shape[1]])
    plt.xlabel("Feature")
    plt.ylabel("Relative importance")
    
    plt.show()
    
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def calculate_frac_ignitions(data, ignition, data_type, num_bins):
    max_val = max(data[data_type])
    min_val = min(data[data_type])
    freq = (max_val - min_val) / num_bins
    bins = pd.interval_range(start=min_val, freq=freq, end=max_val)
    ignitions = pd.cut(ignition[data_type], bins=bins)
    all_data = pd.cut(data[data_type], bins=bins)
    fraction_ignitions = ignitions.value_counts() / all_data.value_counts()
    real_bin_nums = range(len(fraction_ignitions))
    return(fraction_ignitions, real_bin_nums)

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