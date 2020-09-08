import tensorflow as tf
from tensorflow import keras


def predict(data: 'numpy.ndarray', trained_model: str) -> 'numpy.ndarray':
    '''Takes data, pretrained model weights and optimized hyperparameters,
    uses parallel LSTM neural net to predict fire risk. Returns 3D numpy array
    containing predictions by day and location'''

    # ingest weather data, format as list
    input_data = []
    for spatial_bin in data:
        input_data.append(spatial_bin)

    print("input_data is {}, length: {}, member shape: {}.".format(
        type(input_data), len(input_data), input_data[0].shape))

    production_model = tf.keras.models.load_model(trained_model, compile=False)
    production_model.compile()

    # Do prediction
    predictions = production_model.predict(input_data)

    # return numpy ndarray containing predictions
    return predictions
