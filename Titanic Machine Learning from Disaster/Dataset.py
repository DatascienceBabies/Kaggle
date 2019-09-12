import pandas as pd
import math
import numpy as np

class Dataset:
    # Contains the unmodified pandas dataset after being loaded
    dataset = pd.DataFrame()

    # Contains the pandas training parameters
    X = pd.DataFrame()

    # Contains the pandas solutions to the training data X
    Y = pd.DataFrame()

    def __init__(self):
        self.dataset = pd.DataFrame()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

    # Loads a dataset from csv into dataset
    def load_dataset_from_csv(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def get_dataset_parameter(self, parameter_name):
        return self.dataset[parameter_name]

    # Gets the names of available dataset parameters
    def get_dataset_parameter_names(self):
        return self.dataset.columns

    # Generates more X, Y examples to balance Y ratios in classification problems
    # TODO: Incompatible with pandas, fixxy!
    def generate_balanced_data(self):
        sm = SMOTE(random_state=2)
        self.X, self.Y = sm.fit_sample(self.X, self.Y.ravel())

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