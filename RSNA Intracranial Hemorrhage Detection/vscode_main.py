#!/usr/bin/env python
# coding: utf-8

# In[2]:
#%matplotlib inline
#import tensorflow as tf
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import BatchNormalizationV2, Dense, Flatten, Convolution2D, GlobalAveragePooling2D, Input, AveragePooling2D, Activation, Dropout
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
import keras.metrics
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
import sys
import time
import Data_Generator
import BatchDataset as bds
import time
from matplotlib import pyplot as plt
import yaml
import tensorflow as tf
import json


#%% Open the configuration file
with open("config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

    if (os.path.isfile('config.user.yml')):
        with open("config.user.yml", 'r') as user_config_file:
            user_yaml = yaml.load(user_config_file)
            for area in user_yaml:
                for key in user_yaml[area]:
                    print("User override found for: {0}".format(key))
                    config[area][key] = user_yaml[area][key]


#%% Configure the network training
address = config['distributed_training']['address']
task_type = config['distributed_training']['task_type']
task_index = config['distributed_training']['task_index']
total_worker_count = config['distributed_training']['total_worker_count']

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        task_type: [address]
    },
    'task': {'type': task_type, 'index': task_index}
})

# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


# In[3]: Creates a live plot which is shown while a cell is being run
def moving_average(values, averaging_size=3) :
    ret = np.cumsum(values, dtype=float)
    ret[averaging_size:] = ret[averaging_size:] - ret[:-averaging_size]
    return ret[averaging_size - 1:] / averaging_size

def live_plot(data_dict, figsize=(15,5), title='', logarithmic = False, averaging_size=50):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        if logarithmic:
            plt.yscale("log")
        plt.plot(moving_average(data, averaging_size=averaging_size), label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right

    ax_label = plt.axes()

    train_min = data_dict['train_loss'][np.argmin(data_dict['train_loss'])]
    train_min_x = np.argmin(data_dict['train_loss'])
    ax_label.text(
        train_min_x / np.size(data_dict['train_loss']),
        0.01,
        "%.4f" % train_min,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax_label.transAxes,
        color='green', fontsize=15)

    validation_min = data_dict['validation_loss'][np.argmin(data_dict['validation_loss'])]
    validation_min_x = np.argmin(data_dict['validation_loss'])

    ax_label.text(
        validation_min_x / np.size(data_dict['validation_loss']),
        0.01,
        "%.4f" % validation_min,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax_label.transAxes,
        color='green', fontsize=15)

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
        # TODO: Check content size of files to make sure we have the same amount of samples
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
batch_size = config['dataset']['batch_size']
image_width = config['dataset']['image_width']
image_height = config['dataset']['image_height']
train_samples = config['dataset']['train_samples']
test_samples = config['dataset']['test_samples']
target_type = config['dataset']['target_type']
# Warning: Modifications to Data_Generator requires you to remake the cache for it to take effect!
load_existing_cache = config['dataset']['load_existing_cache']
train_cache_file = config['dataset']['train_cache_file']
test_cache_file = config['dataset']['test_cache_file']
train_base_image_path = config['dataset']['train_base_image_path']
test_base_image_path = config['dataset']['test_base_image_path']
use_cache_train = config['dataset']['use_cache_train']
use_cache_test = config['dataset']['use_cache_test']
output_test_images = config['dataset']['output_test_images']
random_train_image_transformation = config['dataset']['random_train_image_transformation']

train_csv_file, test_csv_file = create_specialized_csv(target_type, train_samples, test_samples, load_existing_cache)

batch_dataset_train = bds.BatchDataset('./' + train_csv_file, batch_size)
data_generator_train = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_train,
    image_width,
    image_height,
    train_base_image_path,
    include_resized_mini_images=True,
    output_test_images=output_test_images,
    cache_data=use_cache_train,
    cache_location=train_cache_file,
    keep_existing_cache=load_existing_cache,
    queue_workers=3,
    queue_size=50,
    color=True,
    random_image_transformation=random_train_image_transformation)

batch_dataset_test = bds.BatchDataset('./' + test_csv_file, batch_size)
data_generator_test = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_test,
    image_width,
    image_height,
    test_base_image_path,
    include_resized_mini_images=True,
    cache_data=use_cache_test,
    cache_location=test_cache_file,
    keep_existing_cache=load_existing_cache,
    queue_workers=3,
    queue_size=50,
    color=True)


# In[20]: create model
#with strategy.scope():
#base_model = tf.keras.applications.MobileNetV2(input_shape=(data_generator_train.image_height, data_generator_train.image_width, 1),
#                                               include_top=False,
#                                               weights='imagenet')

#base_model = tf.keras.applications.ResNet50()
base_model = tf.compat.v2.keras.applications.ResNet50(
    include_top=False,
    input_shape=(data_generator_train.image_height, data_generator_train.image_width, 3)
)
base_model.trainable = True

model = Sequential()
model.add(base_model)

#Final decisions
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(80))
model.add(Activation("relu"))
model.add(Dense(240))
model.add(Activation("relu"))
model.add(Dense(80))
model.add(Activation("relu"))
model.add(Dense(35))
model.add(Activation("relu"))

model.add(Dense(2))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.000010)
model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

print(model.summary())

# checkpoint
filepath="weights.{0}.best.hdf5".format(target_type)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
best_val_loss = sys.float_info.max
plotData = collections.defaultdict(list)
model.save('model_{0}.dlm'.format(target_type))

load_existing_weights = config['model']['load_existing_weights']
if load_existing_weights:
    model.load_weights('best_model_weights_{0}.dlm'.format(target_type))

start = time.time()
plotData = collections.defaultdict(list)

def validate_model(model, data_generator):
    validation_loss = 0
    for i in range(data_generator.__len__()):
        validation_loss = validation_loss + model.evaluate(data_generator.getitem_tensorflow_2_X(), data_generator.getitem_tensorflow_2_Y(), verbose=0)[0]
    validation_loss = validation_loss / data_generator.__len__()
    return validation_loss

# In[21]:
# Train the model
epochs_to_train = config['model']['epochs_to_train']
items_trained_per_epoch = config['model']['items_trained_per_epoch']
epochs_between_testing = config['model']['epochs_between_testing']

if items_trained_per_epoch <= 1:
    items_trained_per_epoch = batch_size * batch_dataset_train.batch_amount * items_trained_per_epoch

display_train_loss = config['graph']['display_train_loss']
display_validation_loss = config['graph']['display_validation_loss']
display_train_accuracy = config['graph']['display_train_accuracy']
display_validation_accuracy = config['graph']['display_validation_accuracy']
validation_loss = 0

for i in range(epochs_to_train):
    steps_per_epoch_size = math.floor(items_trained_per_epoch / batch_size)

    history = model.fit(
        x=data_generator_train.getitem_tensorflow_2_X(),
        y=data_generator_train.getitem_tensorflow_2_Y(),
        steps_per_epoch=steps_per_epoch_size,
        epochs=1)

    if i == 0 or i % epochs_between_testing == 0:
        validation_loss = validate_model(model, data_generator_test)

    train_accuracy = history.history['acc'][0]
    train_loss = history.history['loss'][0]
    
    if display_train_loss:
        plotData['train_loss'].append(train_loss)
    if display_validation_loss:
        plotData['validation_loss'].append(validation_loss)
    #if display_train_accuracy:
        #plotData['train_accuracy'].append(train_accuracy)
    #if display_validation_accuracy:
        #plotData['validation_accuracy'].append(validation_accuracy)    

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss
        model.save_weights('best_model_weights_{0}.dlm'.format(target_type))
        print("New best test loss!")
        live_plot(plotData, logarithmic=True, averaging_size=50)
        print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))

    if time.time() - start > 60:
        start = time.time()
        live_plot(plotData, logarithmic=True, averaging_size=50)
        print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))


#%%
