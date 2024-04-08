import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
        optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate = learning_rate),
        loss = "mean_squared_error",
        metrics = [tf.keras.metrics.RootMeanSquaredError()]
    )

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """
    Train model by feeding data
    """
    history = model.fit(
        x = df[feature],
        y = df[label],
        epochs = epochs,
        batch_size = batch_size
    )

    # Gather the model's trained weight and bias
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def train_model_with_validation(model, df, feature, label, epochs, batch_size, validation_split):
    """
    Train model by feeding data. Includes extra parameter to partition training set into validation set
    """
    history = model.fit(
        x = df[feature],
        y = df[label],
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split
    )

    # Gather the model's trained weight and bias
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse, history.history


def predict_from_random(model, df, feature, n):
    """
    Predict label from random feature data
    """

    batch = np.random.uniform(
        df[feature].quantile(.1), 
        df[feature].quantile(.9), 
        size=n)
    
    predicted_values = model.predict_on_batch(x = batch)
    
    output_df = pd.DataFrame()

    for i in range(n):
        row = {
            "Generated feature data": batch[i],
            "Predicted label": predicted_values[i][0]
        }

        output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

    return output_df


def plot_the_model(df, trained_weight, trained_bias, feature, label):
    """
    Plot the trained model against the training feature and label
    """

    # Label the axes
    plt.xlabel("feature")
    plt.ylabel("label")
    
    # Create a scatter plot from 200 random points of the dataset.
    random_examples = df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1)
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """
    Plot the loss curve, which shows loss vs. epoch
    """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


def plot_the_loss_curve_with_validation(epochs, mae_training, mae_validation):
    """
    Plot the loss curve for the training set and the validation set
    """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()