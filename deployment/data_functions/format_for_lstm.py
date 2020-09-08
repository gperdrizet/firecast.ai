import numpy as np
import pandas as pd


def format_data(input_data: 'DataFrame', shape_parameters: dict) -> 'numpy.ndarray':
    '''takes dataframe and information about desired output shape (history window size,
    target window size, timesteps to slide history window) and returns a formatted list'''

    # get nessecary shape parameters into named varible for ease of use
    history_size = shape_parameters['history_size']
    target_size = shape_parameters['target_size']
    step = shape_parameters['step']

    print(input_data.info())

    # empty dataframe to hold samples
    data_list = []

    # break dataframe into spatial bins by lat lon
    spatial_bins = input_data.groupby(['lat', 'lon'])

    # loop on spatial bins and split each into test and training sets
    for bin_name, spatial_bin in spatial_bins:

        # sort bin by date and convert to numpy array
        spatial_bin = spatial_bin.sort_values('date')
        spatial_bin.drop('date', axis=1, inplace=True)
        spatial_bin.drop(['raw_lat', 'raw_lon'], axis=1, inplace=True)
        spatial_bin = np.array(spatial_bin.values)

        bin_data = []

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

        # add to master
        data_list.append(np.array(bin_data))

    return data_list
