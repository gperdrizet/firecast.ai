import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from descartes import PolygonPatch
import helper_functions.data_functions as data_functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def density_plot(ax, plot_location, data, data_type, title, xlabel, ylabel):
    values, base = np.histogram(data[data_type], bins=40)
    values = values / len(data)

    ax[plot_location].plot(base[:-1], values, color='black', linewidth=1)
    ax[plot_location].tick_params(labelsize=12)
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_xlabel(xlabel, fontsize=14)
    ax[plot_location].set_ylabel(ylabel, fontsize=15)
    ax[plot_location].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #ax[plot_location].set_ylim([-0.05,1.05])
    
    return ax

def map_plot(ax, plot_location, california_land_mass, xi, yi, zi, title, num_contour_levels):
    ax[plot_location].add_patch(PolygonPatch(california_land_mass, fc='none', ec='black', lw='2', zorder=2))
    ax[plot_location].contourf(xi, yi, zi, num_contour_levels, cmap='viridis')
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_aspect('equal', adjustable='box')
    return ax

def boxplot(ax, plot_location, no_ignition, ignition, data_type, title, xlabel, ylabel):
    plot_data = [no_ignition[data_type], ignition[data_type]]

    ax[plot_location].boxplot(plot_data, widths = 0.6, patch_artist = False)
    ax[plot_location].tick_params(labelsize=12)
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_xlabel(xlabel, fontsize=14)
    ax[plot_location].set_ylabel(ylabel, fontsize=15)
    ax[plot_location].set_xticklabels(['no','yes'])
    
    return ax
    
def binned_scatterplot(ax, data, ignition, plot_location, data_type, title, xlabel, ylabel, num_bins):
    plot_data, real_bin_nums = data_functions.calculate_frac_ignitions(data, ignition, data_type, num_bins)

    ax[plot_location].plot(real_bin_nums, plot_data, color='black')
    ax[plot_location].tick_params(labelsize=12)
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_xlabel(xlabel, fontsize=14)
    ax[plot_location].set_xticks([])
    ax[plot_location].set_ylabel(ylabel, fontsize=15)
    ax[plot_location].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[plot_location].set_ylim([-0.05,0.125])
    
    return ax

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

def data_diagnostic_plot(data, var, y_streach, y_scale):
    plt.plot(
        data.index, 
        (data['ignition'] * y_streach) + y_scale,
        color = "darkred",
        label ='True ignitions'
    )
    plt.plot(
        data.index, 
        data[var],
        color = "darkgray",
        label ='Air temp'
    )

    plt.xlabel('Day')
    plt.ylabel('Air temp. (K)')
    plt.title('Ignition vs mean air temperature')
    plt.legend()
    plt.xlim(0,365)
    plt.tight_layout()
    plt.show()
    
def one_sample_density_plot(
    ax,
    plot_location, 
    data,  
    data_type, 
    title, 
    xlabel, 
    ylabel, 
):
    values, base = np.histogram(data[data_type], bins=40)

    ax[plot_location].plot(base[:-1], (values/len(data)))
    ax[plot_location].tick_params(labelsize=12)
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_xlabel(xlabel, fontsize=14)
    ax[plot_location].set_ylabel(ylabel, fontsize=15)
    ax[plot_location].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def three_sample_density_plot(
    ax,
    plot_location, 
    sample_1_data,
    sample_2_data,
    sample_3_data,
    data_type, 
    title, 
    xlabel, 
    ylabel, 
):
    sample_1_values, sample_1_base = np.histogram(sample_1_data[data_type], bins=40)
    sample_2_values, sample_2_base = np.histogram(sample_2_data[data_type], bins=40)
    sample_3_values, sample_3_base = np.histogram(sample_3_data[data_type], bins=40)

    ax[plot_location].plot(sample_1_base[:-1], (sample_1_values/len(sample_1_data)))
    ax[plot_location].plot(sample_2_base[:-1], (sample_2_values/len(sample_2_data)))
    ax[plot_location].plot(sample_3_base[:-1], (sample_3_values/len(sample_1_data)))
    ax[plot_location].tick_params(labelsize=12)
    ax[plot_location].set_title(title, fontsize=18)
    ax[plot_location].set_xlabel(xlabel, fontsize=14)
    ax[plot_location].set_ylabel(ylabel, fontsize=15)
    ax[plot_location].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
def plot_metrics(history, metrics, filename):
    '''takes history from model.fit() keras call, training metrics of intrest and filename.
    plots training metrics over time and saves .png figuresd to filename'''
    
    plt.subplots(2, 3, figsize=(10.5, 7.0))
    count_based_metrics = {'true_positives', 'true_negatives', 'false_positives', 'false_negatives'}
    
    # loop on metrics to make plots
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 3, n + 1)
        plt.plot(history.epoch, history.history[metric], color = 'royalblue', label = 'Train')
        plt.plot(history.epoch, history.history['val_' + metric],
             color = 'royalblue', linestyle = "--", label = 'Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(name)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

def plot_ignition_predictions(predictions, actual, filename):
    plt.subplots(1, 4, figsize=(14,4))

    plt.subplot(1, 4, 1)

    plt.plot(
        range(len(actual)), 
        actual,
        color = "darkred",
        label ='True ignitions'
    )
    plt.plot(
        range(len(actual)), 
        predictions,
        color = "darkgray",
        label ='predicted ignitions'
    )

    plt.xlabel('Day')
    plt.ylabel('Ignition')
    plt.title('Predicted vs. actual ignition')
    plt.legend()
    plt.xlim(240,280)

    plt.subplot(1, 4, 2)

    plt.plot(
        range(len(actual)), 
        actual,
        color = "darkred",
        label ='True ignitions'
    )
    plt.plot(
        range(len(actual)), 
        predictions,
        color = "darkgray",
        label = 'predicted ignitions'
    )

    plt.xlabel('Day')
    plt.ylabel('Ignition')
    plt.title('Predicted vs. actual ignition')
    plt.legend()
    plt.xlim(280,320)

    plt.subplot(1, 4, 3)

    plt.plot(
        range(len(actual)), 
        actual,
        color = "darkred",
        label ='True ignitions'
    )
    plt.plot(
        range(len(actual)), 
        predictions,
        color = "darkgray",
        label ='predicted ignitions'
    )

    plt.xlabel('Day')
    plt.ylabel('Ignition')
    plt.title('Predicted vs. actual ignition')
    plt.legend()
    plt.xlim(320,360)

    plt.subplot(1, 4, 4)

    plt.plot(
        range(len(actual)), 
        actual,
        color = "darkred",
        label = 'True ignitions'
    )
    plt.plot(
        range(len(actual)), 
        predictions,
        color = "darkgray",
        label = 'predicted ignitions'
    )

    plt.xlabel('Day')
    plt.ylabel('Ignition')
    plt.title('Predicted vs. actual ignition')
    plt.legend()
    #plt.xlim(1,450)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.tight_layout()
    
def plot_cm(training_labels, training_predictions, test_labels, test_predictions, filename, p=0.5):
    training_cm = confusion_matrix(training_labels, training_predictions > p)
    training_normalized_cm = np.empty([2, 2])
    training_normalized_cm[0][0] = training_cm[0][0] / (training_cm[0][0] + training_cm[0][1])
    training_normalized_cm[0][1] = training_cm[0][1] / (training_cm[0][0] + training_cm[0][1])
    training_normalized_cm[1][0] = training_cm[1][0] / (training_cm[1][0] + training_cm[1][1])
    training_normalized_cm[1][1] = training_cm[1][1] / (training_cm[1][0] + training_cm[1][1])
    
    test_cm = confusion_matrix(test_labels, test_predictions > p)
    test_normalized_cm = np.empty([2, 2])
    test_normalized_cm[0][0] = test_cm[0][0] / (test_cm[0][0] + test_cm[0][1])
    test_normalized_cm[0][1] = test_cm[0][1] / (test_cm[0][0] + test_cm[0][1])
    test_normalized_cm[1][0] = test_cm[1][0] / (test_cm[1][0] + test_cm[1][1])
    test_normalized_cm[1][1] = test_cm[1][1] / (test_cm[1][0] + test_cm[1][1])
    
    plt.subplots(1, 2, figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(training_normalized_cm, annot=True, cmap=("Blues"))
    plt.title('Training confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(test_normalized_cm, annot=True, cmap=("Blues"))
    plt.title('Test data confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print()
    print('Training data prediction results')
    print('No fire (True Negatives): ', training_cm[0][0])
    print('False alarms (False Positives): ', training_cm[0][1])
    print('Fires missed (False Negatives): ', training_cm[1][0])
    print('Fires detected (True Positives): ', training_cm[1][1])
    print('Total fires: ', np.sum(training_cm[1]))
    print()
    print('Test data prediction results')
    print('No fire (True Negatives): ', test_cm[0][0])
    print('False alarms (False Positives): ', test_cm[0][1])
    print('Fires missed (False Negatives): ', test_cm[1][0])
    print('Fires detected (True Positives): ', test_cm[1][1])
    print('Total fires: ', np.sum(test_cm[1]))
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
 