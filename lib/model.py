import tensorflow as tf


def create_model(inputs, labels, normalize=False, epochs=200):

    # Define a model for linear regression.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(2,), name="xy"),
        tf.keras.layers.Dense(4096, name="dense_1"),
        tf.keras.layers.Dense(2, activation='linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Assume `inputs` and `labels` are lists or numpy arrays here.
    # In TensorFlow.js, there seems to be a conversion from some kind of object to an array.
    # So, the exact transformation might need slight adjustments based on your specific data.
    input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

    input_max = tf.reduce_max(input_tensor)
    input_min = tf.reduce_min(input_tensor)
    label_max = tf.reduce_max(label_tensor)
    label_min = tf.reduce_min(label_tensor)

    normalized_inputs = (input_tensor - input_min) / (input_max - input_min)
    normalized_labels = (label_tensor - label_min) / (label_max - label_min)

    # Train the model using the data.
    # Assume `Settings.normalize` and `epochs` are defined elsewhere in your code.
    if normalize:
        model.fit(normalized_inputs, normalized_labels, epochs=epochs)
    else:
        model.fit(input_tensor, label_tensor, epochs=epochs)

    return model