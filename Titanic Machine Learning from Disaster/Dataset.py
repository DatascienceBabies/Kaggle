import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    # Contains the unmodified pandas dataset after being loaded
    dataset = pd.DataFrame()

    # Contains the pandas training parameters
    X = pd.DataFrame()

    # Contains the pandas solutions to the training data X
    Y = pd.DataFrame()

    # True is this is training data with Y, false if this is prediction data
    isTrainData = False

    def __init__(self, isTrainData):
        self.dataset = pd.DataFrame()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.isTrainData = isTrainData

    # Loads a dataset from csv into dataset
    def load_dataset_from_csv(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def get_dataset_parameter(self, parameter_name):
        return self.dataset[parameter_name]

    # Gets the names of available dataset parameters
    def get_dataset_parameter_names(self):
        return self.dataset.columns

    # Returns the X train and test sets according to the provided test ratio (0 - 1)
    def get_X_train_test_sets(self, test_set_ratio):
        testRange = math.floor(self.X.shape[0] * test_set_ratio)
        test_x = self.X.values[range(testRange)]
        train_x = self.X.values[testRange:]
        return train_x, test_x

    # Returns the Y train and test sets according to the provided test ratio (0 - 1)
    def get_Y_train_test_sets(self, test_set_ratio):
        testRange = math.floor(self.X.shape[0] * test_set_ratio)
        test_y = self.Y.values[range(testRange)]
        train_y = self.Y.values[testRange:]
        return train_y, test_y

    def get_train_test_set(self, test_set_ratio, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=test_set_ratio, random_state=random_state)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        return x_train, x_test, y_train, y_test

    def find_null_columns(self):
        null_columns=self.dataset.columns[self.dataset.isnull().any()]
        return self.dataset[null_columns].isnull().sum()