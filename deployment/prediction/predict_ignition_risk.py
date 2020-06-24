import tensorflow as tf
from tensorflow import keras


def predict(data: 'numpy.ndarray', hyperparameters: dict, saved_weights: str) -> 'numpy.ndarray':
    '''Takes data, pretrained model weights and optimized hyperparameters,
    uses parallel LSTM neural net to predict fire risk. Returns 3D numpy array
    containing predictions by day and location'''

    # ingest weather data, format as list
    input_data = []
    for spatial_bin in data:
        input_data.append(spatial_bin)

    print("input_data is {}, length: {}, member shape: {}.".format(
        type(input_data), len(input_data), input_data[0].shape))

    # load hyperparameters into named variables for ease of use
    batch_size = hyperparameters['batch_size']
    lstm_units = hyperparameters['lstm_units']
    variational_dropout = hyperparameters['variational_dropout']
    hidden_units = hyperparameters['hidden_units']
    hidden_l1_lambda = hyperparameters['hidden_l1_lambda']
    output_bias = tf.keras.initializers.Constant(
        hyperparameters['output_bias'])

    # build model
    inputs = []
    LSTMs = []

    # Make one input for each geospatial bin in the data
    for i in range(len(input_data)):
        inputs.append(
            keras.Input(
                batch_shape=(
                    batch_size,
                    input_data[0].shape[1],
                    input_data[0].shape[2]
                )
            )
        )

    # make one parallel LSTM for each geospatial bin
    for i in range(len(input_data)):
        LSTMs.append(keras.layers.LSTM(
            lstm_units,
            dropout=variational_dropout,
            stateful=True
        )(inputs[i]))

    # merge parallel LSTM layers
    merged = keras.layers.concatenate(LSTMs)

    # add a sequence of two hidden fully connected layers
    hidden1 = keras.layers.Dense(
        hidden_units,
        kernel_regularizer=keras.regularizers.l1(hidden_l1_lambda),
        bias_initializer=tf.keras.initializers.he_normal(),
        activation='relu'
    )(merged)

    hidden2 = keras.layers.Dense(
        hidden_units,
        kernel_regularizer=keras.regularizers.l1(hidden_l1_lambda),
        bias_initializer=tf.keras.initializers.he_normal(),
        activation='relu'
    )(hidden1)

    # setup output layer
    output = keras.layers.Dense(
        410,
        activation='sigmoid',
        bias_initializer=output_bias
    )(hidden2)

    # assemble model and load in pre-trained weights
    production_model = keras.Model(
        inputs=inputs, outputs=output)
    production_model.load_weights(saved_weights).expect_partial()

    # Do prediction
    predictions = production_model.predict(input_data)

    # return numpy ndarray containing predictions
    return predictions
