import tensorflow as tf
import pandas as pd

def build_lr_model(learning_rate):
    """
    Create and compile a simple linear regression model
    """

    # First create an instance of a sequential model
    model = tf.keras.models.Sequential()

    # The topography of a linear regression model must be one node in one layer
    model.add(
        tf.keras.layers.Dense(units = 1, input_shape = (1,))
    )
    
    # Compile model topography and configure training to minimize mean squared error
    model.compile(
        optimizer = tf.keras.optimizers.experimental.RSprop(learning_rate = learning_rate),
        loss = "mean_squared_error",
        metrics = [tf.keras.metrics.RootMeanSquaredError()]
    )

    return model

def train_model(model, features, label, epochs, batch_size):
    """
    Train model by feeding data
    """
    history = model.fit(
        x = features,
        y = label,
        epochs = epochs,
        batch_size = batch_size
    )

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs = history.epoch

    hist = pd.DataFrame(history.history)

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse