import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import math
import numpy as np

# TODO: Add tests
# Stores a combination of Dataset modifications which can be applied to multiple Datasets
class DatasetModifier:
    _X_modifiers = []
    _Y_modifiers = []

    def _add_X_parameter(self, dataset, parameter_name):
        dataset.X[parameter_name] = dataset.dataset[parameter_name]

    def _randomize_dataset(self, dataset):
        dataset.dataset = dataset.dataset.sample(frac=1).reset_index(drop=True)        

    def _add_Y_parameter(self, dataset, parameter_name):
        dataset.Y[parameter_name] = dataset.dataset[parameter_name]

    def _one_hot_X_parameter(self, dataset, parameter_name):
        col = dataset.X.loc[:, parameter_name]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(col.values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        dataset.X = dataset.X.drop([parameter_name], axis=1)

        for i in range(0, onehot_encoded.shape[1]):
            new_col_name = "{0}_{1}".format(parameter_name, str(i))
            dataset.X[new_col_name] = onehot_encoded[:, i]

    def _standardize_X(self, dataset):
        numpyX = dataset.X.values #returns a numpy array
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(numpyX)
        dataset.X = pd.DataFrame(x_scaled, columns=dataset.X.columns)


    # Adds a parameter from dataset to X
    def add_X_parameter(self, parameter_name):
        self._X_modifiers.append([self._add_X_parameter, parameter_name])

    # Randomizes the order of the dataset
    def randomize_dataset(self):
        self._X_modifiers.append([self._randomize_dataset])

    # Adds a parameter from dataset to Y
    def add_Y_parameter(self, parameter_name):
        self._Y_modifiers.append([self._add_Y_parameter, parameter_name])

    # One hots a specific parameter in X and drops the original parameter
    def one_hot_X_parameter(self, parameter_name):
        self._X_modifiers.append([self._one_hot_X_parameter, parameter_name])

    # Standardizes the parameter ranges within X
    def standardize_X(self):
        self._X_modifiers.append([self._standardize_X])

    def generate_X(self, dataset):
        for X_modifier in self._X_modifiers:
            # TODO: This can probably be made better
            if len(X_modifier) == 1:
                X_modifier[0](dataset)
            elif len(X_modifier) == 2:
                X_modifier[0](dataset, X_modifier[1])
            elif len(X_modifier) == 3:
                X_modifier[0](dataset, X_modifier[1], X_modifier[2])
            elif len(X_modifier) == 4:
                X_modifier[0](dataset, X_modifier[1], X_modifier[2], X_modifier[3])

    def generate_Y(self, dataset):
        for Y_modifier in self._Y_modifiers:
            # TODO: This can probably be made better
            if len(Y_modifier) == 1:
                Y_modifier[0](dataset)
            elif len(Y_modifier) == 2:
                Y_modifier[0](dataset, Y_modifier[1])
            elif len(Y_modifier) == 3:
                Y_modifier[0](dataset, Y_modifier[1], Y_modifier[2])
            elif len(Y_modifier) == 4:
                Y_modifier[0](dataset, Y_modifier[1], Y_modifier[2], Y_modifier[3])