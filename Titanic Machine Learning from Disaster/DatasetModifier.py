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
    _Dataset_modifiers = []
    _X_modifiers = []
    _Y_modifiers = []
    _X_scaler = StandardScaler()

    _encoders = {}

    def __init__(self):
        self._Dataset_modifiers = []
        self._X_modifiers = []
        self._Y_modifiers = []
        self._encoders = {}
        self._X_scaler = StandardScaler()

    def _add_X_parameter(self, dataset, parameter_name):
        dataset.X[parameter_name] = dataset.dataset[parameter_name]

    def _dataset_randomize(self, dataset, seed = None):
        if seed == None:
            dataset.dataset = dataset.dataset.sample(frac=1).reset_index(drop=True)
        else:
            dataset.dataset = dataset.dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    def _add_Y_parameter(self, dataset, parameter_name):
        dataset.Y[parameter_name] = dataset.dataset[parameter_name]

    def _one_hot_X_parameter(self, dataset, parameter_name):
        col_values = dataset.X.loc[:, parameter_name]
        values_to_encode = col_values.values.reshape(len(col_values.values), 1)

        if parameter_name not in self._encoders:
            self._encoders[parameter_name] = {}
        
        if 'onehot' not in self._encoders[parameter_name]:
            onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self._encoders[parameter_name]['onehot'] = onehot_encoder.fit(values_to_encode)

        onehot_encoder = self._encoders[parameter_name]['onehot']
        onehot_encoded = onehot_encoder.transform(values_to_encode)
        dataset.X = dataset.X.drop([parameter_name], axis=1)

        for i in range(0, onehot_encoded.shape[1]):
            new_col_name = "{0}_{1}".format(parameter_name, str(i))
            dataset.X[new_col_name] = onehot_encoded[:, i]

    def _standardize_X(self, dataset, standardization_dataset):
        train_numerical_features = list(dataset.X.select_dtypes(include=['int64', 'float64', 'int32']).columns)
        dataset_fit = pd.DataFrame(data = standardization_dataset.X)
        dataset_train = pd.DataFrame(data = dataset.X)
        # TODO: This is always run when it would be enough to run it once. Refactor.
        self._X_scaler.fit(dataset_fit[train_numerical_features])
        dataset.X[train_numerical_features] = self._X_scaler.transform(dataset_train[train_numerical_features])

    def _dataset_fill_missing_value(self, dataset, parameter_name):
        data = dataset.dataset[parameter_name]
        mean = data.mean()
        std = data.std()
        is_null = data.isnull().sum()
        rand_values = np.random.normal(mean, std, (is_null))
        dataset.dataset[parameter_name].values[np.where(data.isnull())[0]] = rand_values

    def _dataset_fill_missing_value_based_on_criteria(self, dataset, parameter_name, filter_criteria):
        if parameter_name in dataset.dataset:
            non_null_dataset = dataset.dataset[dataset.dataset[parameter_name].isnull() == False]
            dataset.dataset[parameter_name] = dataset.dataset.apply(lambda row: self._fill_missing_values_based_on_criteria(row, parameter_name, filter_criteria, non_null_dataset), axis=1)

    def _fill_missing_values_based_on_criteria(self, row, parameter_name, filter_criteria, non_null_dataset):
        row_value = row[parameter_name]
        if not bool(row_value):
            filtered_dataset = non_null_dataset.copy()

            for criteria in filter_criteria:
                row_value = row[criteria]
                filtered_dataset = filtered_dataset[filtered_dataset[criteria] == row_value]
                
            return filtered_dataset[parameter_name].median()
        else:
            try:
                return row[parameter_name]
            except:
                return 0

    def _dataset_add_new_feature_based_on_existing(self, dataset, feature_name, existing_features):
        for existing_feature in existing_features:
            if not existing_feature in dataset.dataset.columns:
                return

        dataset.dataset[feature_name] = dataset.dataset.apply(lambda row: self._add_new_feature(row, feature_name, existing_features), axis=1)

    def _add_new_feature(self, row, feature_name, existing_features):
        value = 0

        for existing_feature in existing_features:
            value += row[existing_feature]

        return value

    def _dataset_add_new_feature_based_on_custom_function(self, dataset, feature_name, custom_function):
        dataset.dataset[feature_name] = dataset.dataset.apply(lambda row: custom_function(row), axis=1)

    def _dataset_categorize_number(self, dataset, parameter_name, categories):
        data = dataset.dataset[parameter_name]
        data_categorized = np.where(data <= math.inf, "          ", '')
        for category in categories:
            category_name = category[0]
            category_min = category[1]
            category_max = category[2]
            data_categorized[np.where((data >= category_min) & (data < category_max))] = category_name
        dataset.dataset = dataset.dataset.drop(parameter_name, axis=1)
        dataset.dataset[parameter_name] = data_categorized

    def _dataset_add_boolean_parameter(self, dataset, boolean_parameter_name, filters):
        boolean_parameter = np.ones((dataset.dataset.shape[0], 1))
        for filter in filters:
            parameter_name = filter[0]
            logical_operator = filter[1]
            value = filter[2]
            if logical_operator == '==':
                boolean_parameter[~(dataset.dataset[parameter_name] == value)] = 0
            elif logical_operator == '!=':
                boolean_parameter[~(dataset.dataset[parameter_name] != value)] = 0
            elif logical_operator == '>':
                boolean_parameter[~(dataset.dataset[parameter_name] > value)] = 0
            elif logical_operator == '>=':
                boolean_parameter[~(dataset.dataset[parameter_name] >= value)] = 0
            elif logical_operator == '<':
                boolean_parameter[~(dataset.dataset[parameter_name] < value)] = 0
            elif logical_operator == '<=':
                boolean_parameter[~(dataset.dataset[parameter_name] <= value)] = 0
            else:
                raise ValueError('Does not support that operator.')
        dataset.dataset[boolean_parameter_name] = boolean_parameter
                
                

    def _X_Y_generate_balanced_data(self, dataset):
        sm = SMOTE(random_state=2)
        X, Y = sm.fit_sample(dataset.X, dataset.Y.values.ravel())
        if dataset.Y.columns.shape[0] != 1:
            raise ValueError('Cannot support multicolumn Y.')

        # TODO: See if we can do better than creating a whole new copy?
        dataset.X = pd.DataFrame(X, columns=dataset.X.columns)
        dataset.Y = pd.DataFrame(Y, columns=dataset.Y.columns)

    def _dataset_remove_all_missing_values(self, dataset, parameter_name):
        dataset.dataset = dataset.dataset.dropna(subset=[parameter_name])

    def _dataset_generic_modification(self, dataset, evaluation):
        exec(evaluation)


    # Categorizes a number column in the dataset
    # categories: [[categoryName, min, max], [categoryName, min, max]]
    def dataset_categorize_number(self, parameter_name, categories):
        self._Dataset_modifiers.append([self._dataset_categorize_number, parameter_name, categories])

    # Creates a boolean parameter based on a combination of many parameter ranges
    # filters: [[filter parameter name, logical operator, value]]
    def dataset_add_boolean_parameter(self, boolean_parameter_name, filters):
        self._Dataset_modifiers.append([self._dataset_add_boolean_parameter, boolean_parameter_name, filters])

    # Adds a parameter from dataset to X
    def add_X_parameter(self, parameter_name):
        self._X_modifiers.append([self._add_X_parameter, parameter_name])

    # Randomizes the order of the dataset
    def dataset_randomize(self):
        self._Dataset_modifiers.append([self._dataset_randomize])

    # Adds a parameter from dataset to Y
    def add_Y_parameter(self, parameter_name):
        self._Y_modifiers.append([self._add_Y_parameter, parameter_name])

    # One hots a specific parameter in X and drops the original parameter
    def one_hot_X_parameter(self, parameter_name):
        self._X_modifiers.append([self._one_hot_X_parameter, parameter_name])

    # Standardizes the parameter ranges within X
    def standardize_X(self, standardization_dataset):
        self._X_modifiers.append([self._standardize_X, standardization_dataset])

    # Fills in a random standard deviation value to any missing value field of a parameter
    def dataset_fill_missing_value(self, parameter_name):
        self._Dataset_modifiers.append([self._dataset_fill_missing_value, parameter_name])

    def dataset_fill_missing_value_based_on_criteria(self, parameter_name, filter_criteria):
        self._Dataset_modifiers.append([self._dataset_fill_missing_value_based_on_criteria, parameter_name, filter_criteria])

    def dataset_add_new_feature_based_on_existing(self, feature_name, existing_features):
        self._Dataset_modifiers.append([self._dataset_add_new_feature_based_on_existing, feature_name, existing_features])

    def dataset_add_new_feature_based_on_custom_function(self, feature_name, custom_function):
        self._Dataset_modifiers.append([self._dataset_add_new_feature_based_on_custom_function, feature_name, custom_function])        

    # Generates more X, Y examples to balance Y ratios in classification problems
    def X_Y_generate_balanced_data(self):
        self._Y_modifiers.append([self._X_Y_generate_balanced_data])

    # Removes all dataset items which has a missing value of a certain parameter name
    def dataset_remove_all_missing_values(self, parameter_name):
        self._Dataset_modifiers.append([self._dataset_remove_all_missing_values, parameter_name])

    # Allows a generic modification to the dataset via the exec() functionality
    # Example: TODO: Add example
    def dataset_generic_modification(self, evaluation):
        self._Dataset_modifiers.append([self._dataset_generic_modification, evaluation])

    def generate_dataset(self, dataset):
        for dataset_modifier in self._Dataset_modifiers:
            # TODO: This can probably be made better
            if len(dataset_modifier) == 1:
                dataset_modifier[0](dataset)
            elif len(dataset_modifier) == 2:
                dataset_modifier[0](dataset, dataset_modifier[1])
            elif len(dataset_modifier) == 3:
                dataset_modifier[0](dataset, dataset_modifier[1], dataset_modifier[2])
            elif len(dataset_modifier) == 4:
                dataset_modifier[0](dataset, dataset_modifier[1], dataset_modifier[2], dataset_modifier[3])

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
        
        if dataset.is_train_data:
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
