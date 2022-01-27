import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# if the dataset is still to be formatted, turn this to true
DATASET_TO_FORMAT = False
TRAINING = False

scaler = MinMaxScaler()


# this function checks for format issues and, if it finds them, resolves and re-write the correct dataset.
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


def split_data(percentage, dates, data):
    threshold = int(dates.shape[0] * percentage)

    training_dates = dates[:threshold]
    training_data = data[:threshold, :]

    test_dates = dates[threshold:]
    test_data = data[threshold:, :]

    return training_dates, training_data, test_dates, test_data


def print_losses(history):
    if history is None:
        return

    # training loss -> indicates how well the model fits the already seen data
    loss = history.history['loss']
    # validation loss -> indicates how well the model predicts new data
    val_loss = history.history['val_loss']

    # if tr_loss >> val_loss : overfitting (no good prediction ability)
    # if tr_loss << val_loss : underfitting (the model is still too simple and generalized->probably needs more training)

    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# todo: try to save also the history when 'restore'
def train(model, X_train, Y_train, command):
    history = None
    if command == 'train':
        checkpoint_path = "training/cp.ckpt"
        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        history = model.fit(X_train, Y_train, epochs=20, batch_size=50, validation_split=0.1, callbacks=[cp_callback])

    if command == 'restore':
        model.load_weights("training/cp.ckpt").expect_partial()

    return history


def run():
    # getting normalized data for training and test values
    dates, data = get_right_data('deliveries/dataset/bitcoin_price_Training - Training.csv')

    training_dates, training_data, test_dates, test_data = split_data(percentage=0.8, dates=dates, data=data)

    X_train = []
    Y_train = []

    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60:i])
        Y_train.append(training_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

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
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = train(model, X_train, Y_train, 'restore')

    print_losses(history)

    part_60_days = training_data[-60:, :]
    inputs = np.append(part_60_days, test_data, axis=0)

    X_test = []
    Y_test = []

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = model.predict(X_test)

    scale = 1 / scaler.scale_[0]
    Y_test = Y_test * scale
    Y_pred = Y_pred * scale

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
