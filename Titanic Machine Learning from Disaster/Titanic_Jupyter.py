#!/usr/bin/env python
# coding: utf-8

# In[2]:
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import csv
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
from matplotlib import pyplot as plt
import collections
from IPython.display import clear_output


# In[3]:


df = pd.read_csv('train.csv')

df_train_y = df['Survived'].values
df_train_x = df.drop(['Survived'], axis=1)
df_predict_x = pd.read_csv('test.csv')

test_passenger_ids = df_predict_x['PassengerId']
test_passenger_ids = np.reshape(test_passenger_ids.values, (df_predict_x.shape[0], 1))

labels = df_train_x.columns.values


# # Remove unused Columns

# In[4]:


def remove_unused_cols(dataset, labels):
    unused_cols = ["PassengerId", "Name", "Cabin", "Embarked", "Ticket"]
    
    for col_name in unused_cols:
        labels = labels[labels != col_name]
        dataset = dataset.drop([col_name], axis=1)
       
    
    return dataset, labels

df_train_x, labels = remove_unused_cols(df_train_x, labels)
df_predict_x, _ = remove_unused_cols(df_predict_x, labels)


# # Preprocessing
# 
# **Fill NaN Values for**
# - Age
# 
# **Apply One Hot Encoding on**
# - Sex

# In[7]:


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

# In[26]:


def one_hot_encode_column(dataset, col_names):
    for col_name in col_names:
        col = dataset.loc[:, col_name]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(col.values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        dataset = dataset.drop([col_name], axis=1)

        for i in range(0, onehot_encoded.shape[1]):
            new_col_name = "{0}_{1}".format(col_name, str(i))
            dataset[new_col_name] = onehot_encoded[:, i]
    
    return dataset

columns_to_one_hot_encode = ['Pclass', 'Sex', 'Age']

df_train_x = one_hot_encode_column(df_train_x, columns_to_one_hot_encode)
df_predict_x = one_hot_encode_column(df_predict_x, columns_to_one_hot_encode)


# # Standardize Data

# In[19]:


scaler = StandardScaler().fit(df_train_x)
train_x = scaler.transform(df_train_x)
predict_x = scaler.transform(df_predict_x)


# # Apply Magic
# 
# **Todo**:
# - k-Fold Cross Validation?



#%% Create a test set
testRange = math.floor(train_x.shape[0] * 0.25)
test_x = train_x[range(testRange)]
test_y = df_train_y[range(testRange)]
train_x = train_x[testRange:]
train_y = df_train_y[testRange:]


# In[20]:


# Define binary classification model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=train_x.shape[1]))
model.add(Dropout(0.01))
model.add(Dense(32))
model.add(Dropout(0.01))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[21]:

def UpdateAccuracyPlot(train_accuracy, test_accuracy):
    x = range(len(test_accuracy))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(x, test_accuracy)
    ax.plot(x, train_accuracy)
    ax.set_title("Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    #plt.show()

def live_plot(data_dict, figsize=(7,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show();


# Train the model, iterating on the data in batches of 32 samples
plotData = collections.defaultdict(list)
for i in range(10):
    model.fit(train_x, train_y, epochs=5, batch_size=32)
    test_accuracy = model.evaluate(test_x, test_y)[1]
    train_accuracy = model.evaluate(train_x, train_y)[1]
    plotData['train_accuracy'].append(train_accuracy)
    plotData['test_accuracy'].append(test_accuracy)

live_plot(plotData)

# In[ ]:


# Run the model against the test data
predict_y = model.predict(predict_x)
predict_y = np.around(predict_y)
predict_y = predict_y.astype(np.integer)


# In[ ]:


# Write our predictions to a csv file
csv_predict = np.concatenate((test_passenger_ids, predict_y), axis=1)
csv_predict = np.concatenate((np.reshape(["PassengerId", "Survived"], (1, 2)), csv_predict))
with open('prediction.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_predict)
csvFile.close()


# In[ ]:




