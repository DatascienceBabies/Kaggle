#!/usr/bin/env python
# coding: utf-8

# In[2]:
# %matplotlib inline

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
from keras.layers import Dense, Conv2D, Flatten
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
import Data_Generator
import BatchDataset as bds


# In[3]: Creates a live plot which is shown while a cell is being run
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

def live_image(image):
    clear_output(wait=True)
    plt.imshow(image)
    plt.show()








#%% Define dataset
batch_size = 2

batch_dataset_train = bds.BatchDataset('./stage_1_train_nice.csv', batch_size)
data_generator_train = Data_Generator.Data_Generator(
    batch_dataset_train, 
    'stage_1_train_images',
    './data/rsna-intracranial-hemorrhage-detection.zip')

"""
data_generator_train = Data_Generator.Data_Generator(
    batch_dataset_train, 
    './data/stage_1_train_images/')
"""

batch_dataset_test = bds.BatchDataset(
    './stage_1_test_nice.csv', batch_size)
data_generator_test = Data_Generator.Data_Generator(
    batch_dataset_test,
    'stage_1_train_images',
    './data/rsna-intracranial-hemorrhage-detection.zip')

"""
data_generator_test = Data_Generator.Data_Generator(
    batch_dataset_test,
    './data/stage_1_train_images/')
"""







# In[20]: create model

model = Sequential()
#add model layers
# TODO: Fix the width and height to be dynamic
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(512,512,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    # TODO: Temporarily reduce validation size to get faster tests while developing
    validation_step_size = 5

    model.fit_generator(generator=data_generator_train,
                        steps_per_epoch=1,
                        epochs=1,
                        verbose=1,
                        validation_data=data_generator_test,
                        #validation_steps=batch_dataset_test.batch_amount(),
                        validation_steps=validation_step_size,
                        use_multiprocessing=False,
                        workers=1,
                        max_queue_size=32)

#%%
