#!/usr/bin/env python
# coding: utf-8

# In[2]:
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
import Dataset as ds
import DatasetModifier as dsm


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
    plt.show();


#%% Dataset generation

# Define the parameter creation steps
datasetModifier = dsm.DatasetModifier()
datasetModifier.randomize_dataset()
datasetModifier.add_Y_parameter('Survived')
datasetModifier.add_X_parameter('Sex')
datasetModifier.one_hot_X_parameter('Sex')
datasetModifier.standardize_X()

# Load the train/test dataset
dataset = ds.Dataset()
dataset.load_dataset_from_csv('train.csv')

# Load the prediction dataset
dataset_prediction = ds.Dataset()
dataset_prediction.load_dataset_from_csv('test.csv')

# Apply the parameter creation steps to the two datasets
datasetModifier.generate_X(dataset)
datasetModifier.generate_Y(dataset)
datasetModifier.generate_X(dataset_prediction)

# Fetch the train and test set data
train_X, test_X = dataset.get_X_train_test_sets(0.2)
train_Y, test_Y = dataset.get_Y_train_test_sets(0.2)

# Fetch the predict set data
predict_X, _ = dataset_prediction.get_X_train_test_sets(0)
test_passenger_ids = dataset_prediction.get_dataset_parameter('PassengerId')
test_passenger_ids = np.reshape(test_passenger_ids.values, (test_passenger_ids.shape[0], 1))

# In[20]: Define binary classification model

model = Sequential()
model.add(Dense(5, activation='relu', input_dim=train_X.shape[1]))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[21]:
# Train the model
plotData = collections.defaultdict(list)
for i in range(50):
    model.fit(train_X, train_Y, epochs=1, batch_size=32)
    test_accuracy = model.evaluate(test_X, test_Y)[1]
    train_accuracy = model.evaluate(train_X, train_Y)[1]
    plotData['train_accuracy'].append(train_accuracy)
    plotData['test_accuracy'].append(test_accuracy)
live_plot(plotData)

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

# Should we check for sibling/parents and use this as a criteria for the random value?
# We could potentially even train a network to guess ages based on the other parameters
def fill_missing_age_fields(dataset):
    ages = dataset["Age"]
    mean_age = np.mean(ages[np.where(np.isnan(ages) == False)[0]])
    dataset["Age"].fillna(mean_age, inplace =True)
    
fill_missing_age_fields(df_train_x)
fill_missing_age_fields(df_predict_x)


#%%

def categorize_ages(dataset):
    ages = dataset["Age"]
    ages_categorized = np.where(ages >= 0, "          ", '')
    ages_categorized[np.where(ages < 2)] = "infant"
    ages_categorized[np.where((ages >= 2) & (ages < 10))] = "child"
    ages_categorized[np.where((ages >= 10) & (ages < 18))] = "teenager"
    ages_categorized[np.where((ages >= 18) & (ages < 30))] = "youngAdult"
    ages_categorized[np.where((ages >= 30) & (ages < 50))] = "midlife"
    ages_categorized[np.where((ages >= 50))] = "oldFart"
    return dataset.assign(Age = lambda x: ages_categorized)
    #dataset.assign()

df_train_x = categorize_ages(df_train_x)
df_predict_x = categorize_ages(df_predict_x)
