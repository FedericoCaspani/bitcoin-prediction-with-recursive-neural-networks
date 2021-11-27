import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# if the dataset is still to be formatted, turn this to true
DATASET_TO_FORMAT = False


# this function checks for format issues and, if it founds them, resolves and re-write the correct dataset.
# otherwise, reads data and returns it in a normalized version.
def get_right_data():

    data = pd.read_csv('deliveries/dataset/bitcoin_price_Training - Training.csv', date_parser=True)
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
        new_dataset.to_csv("deliveries/dataset/bitcoin_price_Training - Training.csv")

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return dates, data


def run():

    test_date, training_data = get_right_data()
    X_train = []
    Y_train = []

    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60:i])
        Y_train.append(training_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # Initialize the RNN
    model = Sequential()
    model.add(LSTM(units=70, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 7)))
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
    history = model.fit(X_train, Y_train, epochs=20, batch_size=50, validation_split=0.1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print(np.__version__)
    run()
