import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# if the dataset is still to be formatted, turn this to true
DATASET_TO_FORMAT = True


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

    print("before: "+str(data))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    print("after: "+str(data))

    return dates, data


def run():

    test_date, training_data = get_right_data()

    print("data type:"+str(type(training_data)))
    print("dates type: "+str(type(test_date)))


if __name__ == '__main__':
    run()
