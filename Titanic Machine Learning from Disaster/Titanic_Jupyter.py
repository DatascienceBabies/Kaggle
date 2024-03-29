#!/usr/bin/env python
# coding: utf-8

# In[2]:
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import csv
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
from matplotlib import pyplot as plt
import collections
from IPython.display import clear_output

import os
import sys
current_dir = os.getcwd()
util_path = os.path.join(os.path.dirname(current_dir), '', 'util')
sys.path.append(util_path)
import Dataset as ds
import DatasetModifier as dsm
from keras.callbacks import ModelCheckpoint
import sys
import time


# In[3]:

# Creates a live plot which is shown while a cell is being run
def live_plot(data_dict, figsize=(15,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()


#%% Dataset generation

def extract_title_from_name(row):
    return row['Name'].split(',')[1].split('.')[0].strip()

# Load the train/test dataset
dataset_orig = ds.Dataset(has_Y_param = True)
dataset_orig.load_dataset_from_csv('train.csv')
dataset_train, dataset_test = dataset_orig.divide_dataset(0.80)
dataset_orig = None

# Load the prediction dataset
dataset_prediction = ds.Dataset(has_Y_param = False)
dataset_prediction.load_dataset_from_csv('test.csv')

# Define the parameter creation steps
datasetModifier = dsm.DatasetModifier()
datasetModifier.dataset_randomize()

# Define the parameter creation steps
datasetModifier.dataset_fill_missing_value_based_on_criteria('Age', ['Sex', 'Pclass'])
#datasetModifier.dataset_categorize_number('Age', [['0', 0, 16], ['1', 16, 32], ['2', 32, 48], ['3', 48, 64], ['4', 64, math.inf]])
datasetModifier.dataset_categorize_number('Age', [['infant', 0, 2], ['child', 2, 10], ['teenager', 10, 18], ['youngadult', 18, 30], ['midlife', 30, 50], ['oldfart', 50, math.inf]])
#datasetModifier.dataset_fill_missing_value('Age')
datasetModifier.add_X_parameter('Age')
datasetModifier.one_hot_X_parameter('Age')

datasetModifier.add_X_parameter('Sex')
datasetModifier.one_hot_X_parameter('Sex')

datasetModifier.dataset_add_new_feature_based_on_existing('FamilySize', ['SibSp', 'Parch'])
datasetModifier.add_X_parameter('FamilySize')

datasetModifier.dataset_add_new_feature_based_on_custom_function('Title', extract_title_from_name)
datasetModifier.add_X_parameter('Title')
datasetModifier.one_hot_X_parameter('Title')

#datasetModifier.dataset_fill_missing_value('Fare')
datasetModifier.dataset_fill_missing_value_based_on_criteria('Fare', ['Sex', 'Pclass', 'Title'])
#datasetModifier.dataset_categorize_number('Fare', [['poor feck', 0, 10], ['middle class', 10, 50], ['richy rich', 50, math.inf]])
datasetModifier.add_X_parameter('Fare')
#datasetModifier.one_hot_X_parameter('Fare')

datasetModifier.dataset_fill_missing_value('Embarked', 'S')
datasetModifier.add_X_parameter('Embarked')
datasetModifier.one_hot_X_parameter('Embarked')

#datasetModifier.standardize_X(dataset)

datasetModifier.add_Y_parameter('Survived')

#datasetModifier.dataset_remove_all_missing_values('Age')
#datasetModifier.dataset_fill_missing_value('Age')
#datasetModifier.dataset_add_boolean_parameter(
#    'boyChild', [['Sex', '==', 'male'], ['Age', '<=', 10]])
#datasetModifier.add_X_parameter('boyChild')
#datasetModifier.one_hot_X_parameter('boyChild')

#datasetModifier.dataset_add_boolean_parameter(
#    'teenMan', [['Sex', '==', 'male'], ['Age', '>', 10], ['Age', '<', 18]])
#datasetModifier.add_X_parameter('teenMan')
#datasetModifier.one_hot_X_parameter('teenMan')

#datasetModifier.dataset_add_boolean_parameter(
#    'grownMan', [['Sex', '==', 'male'], ['Age', '>', 18]])
#datasetModifier.add_X_parameter('grownMan')
#datasetModifier.one_hot_X_parameter('grownMan')

#datasetModifier.dataset_categorize_number('Age', [['child1', 0, 3], ['child2', 3, 6], ['child3', 6, 9], ['child4', 9, 12], ['teenager', 13, 18], ['youngadult', 18, 40], ['midlife', 40, 55], ['oldfart', 55, math.inf]])
#datasetModifier.dataset_categorize_number('Age', [['child', 0, 5], ['adult', 5, math.inf]])
#datasetModifier.add_X_parameter('Age')
#datasetModifier.one_hot_X_parameter('Age')

#datasetModifier.add_X_parameter('Age')

#datasetModifier.add_X_parameter('Pclass')
#datasetModifier.one_hot_X_parameter('Pclass')

#datasetModifier.add_X_parameter('Sex')
#datasetModifier.one_hot_X_parameter('Sex')

datasetModifier.dataset_add_boolean_parameter(
    'ManFromS', [['Sex', '==', 'male'], ['Embarked', '==', 'S']])
datasetModifier.add_X_parameter('ManFromS')
#datasetModifier.one_hot_X_parameter('ManFromS')

datasetModifier.dataset_add_boolean_parameter(
    'ManFromQ', [['Sex', '==', 'male'], ['Embarked', '==', 'Q']])
datasetModifier.add_X_parameter('ManFromQ')
#datasetModifier.one_hot_X_parameter('ManFromQ')

datasetModifier.dataset_add_boolean_parameter(
    'ManFromC', [['Sex', '==', 'male'], ['Embarked', '==', 'C']])
datasetModifier.add_X_parameter('ManFromC')
#datasetModifier.one_hot_X_parameter('ManFromC')

datasetModifier.dataset_add_boolean_parameter(
    'WomanFromS', [['Sex', '==', 'male'], ['Embarked', '==', 'S']])
datasetModifier.add_X_parameter('WomanFromS')
#datasetModifier.one_hot_X_parameter('WomanFromS')

datasetModifier.dataset_add_boolean_parameter(
    'WomanFromQ', [['Sex', '==', 'male'], ['Embarked', '==', 'Q']])
datasetModifier.add_X_parameter('WomanFromQ')
#datasetModifier.one_hot_X_parameter('WomanFromQ')

datasetModifier.dataset_add_boolean_parameter(
    'WomanFromC', [['Sex', '==', 'male'], ['Embarked', '==', 'C']])
datasetModifier.add_X_parameter('WomanFromC')
#datasetModifier.one_hot_X_parameter('WomanFromC')

#datasetModifier.dataset_fill_missing_value('Fare')
#datasetModifier.add_X_parameter('Fare')

#datasetModifier.dataset_fill_missing_value('Embarked', 'S')
#datasetModifier.add_X_parameter('Embarked')
#datasetModifier.one_hot_X_parameter('Embarked')

#datasetModifier.dataset_fill_missing_value('SibSp')
#datasetModifier.add_X_parameter('SibSp')
#datasetModifier.one_hot_X_parameter('SibSp')

#datasetModifier.dataset_fill_missing_value('Parch')
#datasetModifier.add_X_parameter('Parch')
#datasetModifier.one_hot_X_parameter('Parch')

#datasetModifier.dataset_add_new_feature_based_on_custom_function('Title', extract_title_from_name)
#datasetModifier.add_X_parameter('Title')
#datasetModifier.one_hot_X_parameter('Title')

#datasetModifier.standardize_X(dataset_train)

datasetModifier.add_Y_parameter('Survived')

#datasetModifier.X_Y_generate_balanced_data()

# Apply the parameter creation steps to the two datasets
datasetModifier.generate_dataset(dataset_train)
datasetModifier.generate_dataset(dataset_test)
datasetModifier.generate_dataset(dataset_prediction)

# Fetch the train and test set data
train_X = dataset_train.X.values
train_Y = dataset_train.Y.values
test_X = dataset_test.X.values
test_Y = dataset_test.Y.values

# Fetch the predict set data
predict_X = dataset_prediction.X.values
test_passenger_ids = dataset_prediction.get_dataset_parameter('PassengerId')
test_passenger_ids = np.reshape(test_passenger_ids.values, (test_passenger_ids.shape[0], 1))

# In[20]: Define binary classification model

optimizer = Adam(lr=0.00010, decay=0.0005)
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=train_X.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
best_test_loss = sys.float_info.max
plotData = collections.defaultdict(list)
model.save('model')

start = time.time()

# In[21]:
# Train the model
for i in range(60000):
    model.fit(train_X, train_Y, epochs=20, batch_size=train_X.shape[0], verbose=0)
    loss_train = model.evaluate(train_X, train_Y, verbose=0)[0]
    loss_test = model.evaluate(test_X, test_Y, verbose=0)[0]

    accuracy_train = np.sum(train_Y == np.around(model.predict(train_X, verbose=0)).astype(np.bool)) / train_X.shape[0]
    accuracy_test = np.sum(test_Y == np.around(model.predict(test_X, verbose=0)).astype(np.bool)) / test_X.shape[0]

    plotData['loss_train'].append(loss_train)
    plotData['loss_test'].append(loss_test)

    plotData['accuracy_train'].append(accuracy_train)
    plotData['accuracy_test'].append(accuracy_test)

    if best_test_loss > loss_test:
        best_test_loss = loss_test
        model.save_weights('best_model_weights')
        print("New best test loss!")
        live_plot(plotData)
        print("AT:", round(accuracy_test, 5), " LT: ", round(loss_test, 5))

    if time.time() - start > 60:
        start = time.time()
        live_plot(plotData)
        print("AT:", round(accuracy_test, 5), " LT: ", round(loss_test, 5))        

# In[ ]:


# Run the model against the test data
predict_Y = model.predict(predict_X)
predict_Y = np.around(predict_Y)
predict_Y = predict_Y.astype(np.integer)


# In[ ]:


# Write our predictions to a csv file
csv_predict = np.concatenate((test_passenger_ids, predict_Y), axis=1)
csv_predict = np.concatenate((np.reshape(["PassengerId", "Survived"], (1, 2)), csv_predict))
with open('prediction.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_predict)
csvFile.close()
















# In[7]:
# # Preprocessing
# 
# **Fill NaN Values for**
# - Age
# 
# **Apply One Hot Encoding on**
# - Sex


# # Apply Magic
# 
# **Todo**:
# - k-Fold Cross Validation?