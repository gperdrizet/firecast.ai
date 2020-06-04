import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix

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
 