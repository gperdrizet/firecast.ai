import tensorflow as tf
from tensorflow import keras


def predict(data: 'numpy.ndarray', hyperparameters: dict, saved_weights: str) -> 'numpy.ndarray':
    input_data = []
    for spatial_bin in data:
        input_data.append(spatial_bin)

    print("input_data is {}, length: {}, member shape: {}.".format(
        type(input_data), len(input_data), input_data[0].shape))

    batch_size = hyperparameters['batch_size']
    lstm_units = hyperparameters['lstm_units']
    variational_dropout = hyperparameters['variational_dropout']
    hidden_units = hyperparameters['hidden_units']
    hidden_l1_lambda = hyperparameters['hidden_l1_lambda']
    output_bias = tf.keras.initializers.Constant(
        hyperparameters['output_bias'])

    inputs = []
    LSTMs = []

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

    for i in range(len(input_data)):
        LSTMs.append(keras.layers.LSTM(
            lstm_units,
            dropout=variational_dropout,
            stateful=True
        )(inputs[i]))

    merged = keras.layers.concatenate(LSTMs)

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

    output = keras.layers.Dense(
        410,
        activation='sigmoid',
        bias_initializer=output_bias
    )(hidden2)

    production_model = keras.Model(
        inputs=inputs, outputs=output)
    production_model.load_weights(saved_weights).expect_partial()
    print(production_model.summary())
    predictions = production_model.predict(input_data)

    return predictions
