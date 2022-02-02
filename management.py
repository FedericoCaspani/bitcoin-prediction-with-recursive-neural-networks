"""
This file contains all the configuration stuff needed to manage
all the features of the neural network from outside.
"""

DATASET_PATH = 'deliveries/dataset/bitcoin_price_Training.csv'

# True => Gets the dataset, re-formats it and overwrites on the non formatted file path
# False => Reads the file in path and directly uses it
DATASET_TO_FORMAT = False

# True => Training process starts from the beginning
# False => the previously saved knowledge gets restored
TRAINING = True

# Percentage ([0,1]) of the training-test splitting range
SPLIT_PERCENTAGE = 0.8

# Visualization parameters
PRINT_ARCHITECTURE = True
PRINT_LOSSES = True
PRINT_PREDICTION = True
PRINT_FINAL_ACCURACY = True



