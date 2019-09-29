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
from keras.layers import AveragePooling2D
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Convolution2D
from keras import optimizers
import time
import keras.metrics
from matplotlib import pyplot as plt


# In[3]: Creates a live plot which is shown while a cell is being run
def live_plot(data_dict, figsize=(15,5), title='', logarithmic = False):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        if logarithmic:
            plt.yscale("log")
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






#%%
def create_specialized_csv(target_type, train_samples, test_samples, keep_existing_cache):
    train_csv_file = target_type + '_train_' + str(train_samples) + '.csv'
    test_csv_file = target_type + '_test_' + str(test_samples) + '.csv'

    if keep_existing_cache and os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
        # If csv files already exist, then keep them
        return train_csv_file, test_csv_file

    ds = pd.read_csv('stage_1_train_nice.csv')        

    if train_samples + test_samples <= 1:
        total_examples = ds[ds[target_type] == 1].shape[0]
        train_samples = math.floor(total_examples * train_samples)
        test_samples = math.floor(total_examples * test_samples)
    
    dataset = ds[ds[target_type] == 1].sample(train_samples)
    ds = ds.drop(dataset.index)
    none_ds = ds[ds['any'] == 0].sample(train_samples)
    ds = ds.drop(none_ds.index)
    epidural_train_ds = pd.concat([dataset, none_ds]).sample(frac=1)
    epidural_train_ds.to_csv(train_csv_file, index=None, header=True)

    dataset = ds[ds[target_type] == 1].sample(test_samples)
    ds = ds.drop(dataset.index)
    none_ds = ds[ds['any'] == 0].sample(test_samples)
    ds = ds.drop(none_ds.index)
    epidural_test_ds = pd.concat([dataset, none_ds]).sample(frac=1)
    epidural_test_ds.to_csv(test_csv_file, index=None, header=True)

    return train_csv_file, test_csv_file




#%% Define dataset
batch_size = 100
image_width = 128
image_height = 128
train_samples = 0.98
test_samples = 0.02
target_type = 'intraventricular'
# Warning: Modifications to Data_Generator requires you to remake the cache for it to take effect!
load_existing_cache = False

train_csv_file, test_csv_file = create_specialized_csv(target_type, train_samples, test_samples, load_existing_cache)

batch_dataset_train = bds.BatchDataset('./' + train_csv_file, batch_size)
data_generator_train = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_train,
    image_width,
    image_height,
    './data/stage_1_train_images',
    include_resized_mini_images=True,
    output_test_images=False,
    cache_data=True,
    cache_location='./data/cache_train.dat',
    keep_existing_cache=load_existing_cache,
    queue_workers=3,
    queue_size=50)
#data_generator_train = Data_Generator.Data_Generator(batch_dataset_train, image_width, image_height, 'stage_1_train_images', './data/rsna-intracranial-hemorrhage-detection.zip')

batch_dataset_test = bds.BatchDataset('./' + test_csv_file, batch_size)
data_generator_test = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_test,
    image_width,
    image_height,
    './data/stage_1_train_images',
    include_resized_mini_images=True,
    cache_data=True,
    cache_location='./data/cache_test.dat',
    keep_existing_cache=load_existing_cache,
    queue_workers=1,
    queue_size=30)
#data_generator_test = Data_Generator.Data_Generator(batch_dataset_test, image_width, image_height, 'stage_1_train_images', './data/rsna-intracranial-hemorrhage-detection.zip')
#data_generator_test = Data_Generator.Data_Generator(batch_dataset_test, image_width, image_height, './data/stage_1_train_images/')


# In[20]: create model

#model = Sequential()
#add model layers
# TODO: Fix the width and height to be dynamic
#model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(512,512,1)))
#model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, kernel_size=3))
#model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, kernel_size=3))
#model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Dense(128))
#model.add(Dense(128))
#model.add(Dense(1, activation='sigmoid'))

#visible = Input(shape=(512,512,3))
#conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
#pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
#pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#flat = Flatten()(pool2)
#hidden1 = Dense(10, activation='relu')(flat)
#output = Dense(2, activation='sigmoid')(hidden1)
#model = Model(inputs=visible, outputs=output)
# plot graph
#plot_model(model, to_file='convolutional_neural_network.png')

#base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
#x=base_model.output
#x=GlobalAveragePooling2D()(x)
#x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
#x=Dense(1024,activation='relu')(x) #dense layer 2
#x=Dense(512,activation='relu')(x) #dense layer 3
#prediction=Dense(2,activation='softmax')(x) #final layer with softmax activation

#model=Model(inputs=base_model.input,outputs=prediction)

#for layer in model.layers:
#    layer.trainable=False
# Make the pretrained model layer static (no training)
#for layer in model.layers[:len(base_model.layers)]:
#    layer.trainable=False
#for layer in model.layers[len(base_model.layers):]:
#    layer.trainable=True

#optimizer = Adam(lr=0.10000, decay=0.0000)

#compile model using accuracy to measure model performance
#model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode='same',name='conv1_1', input_shape = (data_generator_train.image_height, data_generator_train.image_width, 3)))
#first_layer = model.layers[0]
# this is a placeholder tensor that will contain our generated images
#input_img = first_layer.input
#dream = input_img
model.add(Activation("relu"))
model.add(Dropout(0.01))
model.add(Convolution2D(32, 5, 5, border_mode='same',name='conv1_2'))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.01))
#model.add(Dropout(0.25))

#2
model.add(Convolution2D(64, 5, 5, border_mode='same',name='conv2_1_1'))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.01))
model.add(Convolution2D(64, 5, 5, border_mode='same',name='conv2_2_2'))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.01))
#model.add(Dropout(0.25))

#flatten
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(Dense(300))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Dropout(0.01))
model.add(Activation("relu"))
model.add(Dense(50))
model.add(Dropout(0.01))
model.add(Activation("relu"))

model.add(Dense(2))
model.add(Activation('softmax'))

#rms = optimizers.RMSprop()
#sgd = optimizers.SGD(lr=0.000010, decay=1e-6, momentum=0.5, nesterov=True)
optimizer = Adam(lr=0.00001, decay=0.000001)
model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
#model.fit(Xtrain, Ytrain, batch_size=32, nb_epoch=100,
#          verbose=1)

print(model.summary())

# checkpoint
filepath="weights.{0}.best.hdf5".format(target_type)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
best_val_loss = sys.float_info.max
plotData = collections.defaultdict(list)
model.save('model_{0}'.format(target_type))
model.load_weights('best_model_weights_{0}'.format(target_type))

start = time.time()

plotData = collections.defaultdict(list)

# In[21]:
# Train the model
for i in range(60000):
    # TODO: Temporarily reduce validation size to get faster tests while developing
    steps_per_epoch_size = batch_dataset_train.batch_amount() / 40
    validation_step_size = batch_dataset_test.batch_amount() / 5

    history = model.fit_generator(generator=data_generator_train,
                        steps_per_epoch=steps_per_epoch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=data_generator_test,
                        #validation_steps=batch_dataset_test.batch_amount(),
                        validation_steps=validation_step_size,
                        use_multiprocessing=False,
                        workers=0,
                        max_queue_size=32)

    train_accuracy = history.history['acc'][0]
    validation_accuracy = history.history['val_acc'][0]
    train_loss = history.history['loss'][0]
    validation_loss = history.history['val_loss'][0]
    
    plotData['train_loss'].append(train_loss)
    plotData['validation_loss'].append(validation_loss)
    #plotData['train_accuracy'].append(train_accuracy)
    #plotData['validation_accuracy'].append(validation_accuracy)    

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss
        model.save_weights('best_model_weights_{0}'.format(target_type))
        print("New best test loss!")
        live_plot(plotData, logarithmic=True)
        print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))

    if time.time() - start > 60:
        start = time.time()
        live_plot(plotData, logarithmic=True)
        print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))


#%%
