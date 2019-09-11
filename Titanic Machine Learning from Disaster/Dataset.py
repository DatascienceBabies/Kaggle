import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
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

    # Randomizes the order of the dataset
    def randomize_dataset(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def get_dataset_parameter(self, parameter_name):
        return self.dataset[parameter_name]

    # Adds a parameter from dataset to X
    def add_X_parameter(self, parameter_name):
        self.X[parameter_name] = self.dataset[parameter_name]

    # Adds a parameter from dataset to Y
    def add_Y_parameter(self, parameter_name):
        self.Y[parameter_name] = self.dataset[parameter_name]

    # Gets the names of available dataset parameters
    def get_dataset_parameter_names(self):
        return self.dataset.columns

    # One hots a specific parameter in X and drops the original parameter
    def one_hot_X_parameter(self, parameter_name):
        col = self.X.loc[:, parameter_name]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(col.values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        self.X = self.X.drop([parameter_name], axis=1)

        for i in range(0, onehot_encoded.shape[1]):
            new_col_name = "{0}_{1}".format(parameter_name, str(i))
            self.X[new_col_name] = onehot_encoded[:, i]

    # Standardizes the parameter ranges within X
    def standardize_X(self):
        numpyX = self.X.values #returns a numpy array
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(numpyX)
        self.X = pd.DataFrame(x_scaled, columns=self.X.columns)

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