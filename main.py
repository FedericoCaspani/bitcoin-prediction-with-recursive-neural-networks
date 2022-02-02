from management import *  # importing environmental variables from management.py

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


scaler = MinMaxScaler()


# This function checks for format issues and, if it finds them, resolves and re-write the correct dataset.
# otherwise, reads data and returns it in a normalized version.
def get_right_data(path):
    data = pd.read_csv(path, date_parser=True)
    data.tail()

    ''' copying the date in another variable in order to remove it 
        from training_data and normalize it to a certain type of values '''

    dates = data['Date'].copy()
    data = data.drop(['Date'], axis=1)

    if DATASET_TO_FORMAT:

        # removing non-float characters to apply normalization (s.g. commas, slashes, ...) after #
        chars_to_remove = [',']
        for c in chars_to_remove:
            data.replace(c, '', regex=True, inplace=True)

        # re-writing the csv file
        new_dataset = dates.to_frame().join(data)
        new_dataset.to_csv(path, index=False)

    # flipping data (reversing rows), to have a crescent set of tuples wrt their dates
    data = np.flipud(data)
    dates = np.flipud(dates)

    # scaling data in order to have a normalized dataset to work on
    data = scaler.fit_transform(data)

    return dates, data


def split_data(dates, data):
    if SPLIT_PERCENTAGE < 0 or SPLIT_PERCENTAGE > 1:
        return

    threshold = int(dates.shape[0] * SPLIT_PERCENTAGE)

    training_dates = dates[:threshold]
    training_data = data[:threshold, :]

    test_dates = dates[threshold:]
    test_data = data[threshold:, :]

    return training_dates, training_data, test_dates, test_data


# Construct blocks of size "block_size" shifting by 1 sample each time.
def build_batches(data, block_size, field_to_predict):
    x = []
    y = []
    for i in range(block_size, data.shape[0]):
        x.append(data[i - block_size:i])
        y.append(data[i, field_to_predict])
    x, y = np.array(x), np.array(y)
    return x, y


def print_losses(history):

    if history is None:
        return

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# todo: try to save also the history when 'restore'
def train(model, X_train, Y_train):
    history = None
    if TRAINING:
        checkpoint_path = "saved_knowledge/cp.ckpt"
        # Create a callback that saves the model's weights and biases
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        history = model.fit(X_train, Y_train, epochs=20, batch_size=50, validation_split=0.1, callbacks=[cp_callback])

    else:
        model.load_weights("saved_knowledge/cp.ckpt").expect_partial()

    return history


def calculate_accuracy(y, y_pred):
    error = 0

    for i in range(len(y)):
        error += np.abs(y[i] - y_pred[i]) / y[i] * 100
    error = error / len(y)

    return 100 - error


def run():
    # getting normalized data for training and test values
    dates, data = get_right_data(path=DATASET_PATH)

    training_dates, training_data, test_dates, test_data = split_data(dates=dates, data=data)

    X_train, Y_train = build_batches(data=training_data, block_size=60, field_to_predict=0)

    # Initialize the RNN
    model = Sequential()
    model.add(LSTM(units=70, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 6)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=140, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    if PRINT_ARCHITECTURE:
        model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = train(model, X_train, Y_train)
    if PRINT_LOSSES:
        print_losses(history)

    part_60_days = training_data[-60:, :]
    inputs = np.append(part_60_days, test_data, axis=0)

    X_test, Y_test = build_batches(data=inputs, block_size=60, field_to_predict=0)

    Y_pred = model.predict(X_test)

    scale = 1 / scaler.scale_[0]
    Y_test = Y_test * scale
    Y_pred = Y_pred * scale

    if PRINT_FINAL_ACCURACY:
        accuracy = calculate_accuracy(Y_test, Y_pred)
        print("Accuracy of the prediction: " + str(accuracy[0]) + " % ")

    if PRINT_PREDICTION:
        plt.figure(figsize=(14, 5))
        plt.plot(Y_test, color='red', label='Real Bitcoin Price')
        plt.plot(Y_pred, color='green', label='Predicted Bitcoin Price')

        # setting the division of X-axis, in order to print only some dates and not overload the axis
        date_tick = np.arange(0, len(test_dates) + 1, 50)
        plt.xticks(date_tick, test_dates[date_tick])

        plt.title('Bitcoin Price Prediction using RNN-LSTM')
        plt.xlabel('Time [Day]')
        plt.ylabel('Price [$]')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run()
