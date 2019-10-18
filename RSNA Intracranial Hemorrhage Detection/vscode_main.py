#!/usr/bin/env python
# coding: utf-8

# In[2]:
#%matplotlib inline
#import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, GlobalAveragePooling2D, Input, AveragePooling2D, Activation, Dropout
from tensorflow.keras.callbacks import TensorBoard
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
import logging
from Data_Generator_Cache import Data_Generator_Cache
import datetime




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

data_location = config['dataset']['data_location']




#%%
logging.basicConfig(filename=os.path.join(data_location, 'logOutput.log'),level=logging.DEBUG)


#%% Configure the network training
address = config['distributed_training']['address']
task_type = config['distributed_training']['task_type']
task_index = config['distributed_training']['task_index']
total_worker_count = config['distributed_training']['total_worker_count']

#os.environ['TF_CONFIG'] = json.dumps({
#    'cluster': {
#        task_type: [address]
#    },
#    'task': {'type': task_type, 'index': task_index}
#})

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
        verticalalignment='bottom',
        horizontalalignment='left',
        transform=ax_label.transAxes,
        color='blue',
        fontsize=15)

    validation_min = data_dict['validation_loss'][np.argmin(data_dict['validation_loss'])]
    validation_min_x = np.argmin(data_dict['validation_loss'])

    ax_label.text(
        validation_min_x / np.size(data_dict['validation_loss']),
        0.15,
        "%.4f" % validation_min,
        verticalalignment='bottom',
        horizontalalignment='left',
        transform=ax_label.transAxes,
        color='yellow', fontsize=15)

    plt.show()

def live_image(image):
    clear_output(wait=True)
    plt.imshow(image)
    plt.show()






#%%
def create_specialized_csv(target_type, train_samples, test_samples, keep_existing_cache, data_folder='./data', use_complete_dataset=True):
    train_csv_file = data_folder + '/' + target_type + '_train_' + str(train_samples) + '.csv'
    test_csv_file = data_folder + '/' + target_type + '_test_' + str(test_samples) + '.csv'

    if keep_existing_cache and os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
        # If csv files already exist, then keep them
        # TODO: Check content size of files to make sure we have the same amount of samples
        return train_csv_file, test_csv_file

    ds = pd.read_csv('stage_1_train_nice.csv')
    if test_samples <= 1:
        test_samples = math.floor(ds[ds[target_type] == 1].shape[0] * test_samples)

    if train_samples <= 1:
        if use_complete_dataset:
            # Use the complete dataset. Copy the dataset which is smaller
            if ds[ds[target_type] != 1].shape[0] > ds[ds[target_type] == 1].shape[0]:
                total_examples = ds[ds[target_type] != 1].shape[0]
            else:
                total_examples = ds[ds[target_type] == 1].shape[0]
            train_samples = math.floor(total_examples - test_samples)
        else:
            total_examples = ds[ds[target_type] == 1].shape[0]
            train_samples = math.floor(total_examples * train_samples)

    dataset = ds[ds[target_type] == 1].sample(test_samples)
    ds = ds.drop(dataset.index)
    none_ds = ds[ds[target_type] == 0].sample(test_samples)
    ds = ds.drop(none_ds.index)
    test_ds = pd.concat([dataset, none_ds]).sample(frac=1)
    test_ds.to_csv(test_csv_file, index=None, header=True)        
    
    dataset = ds[ds[target_type] == 1].sample(train_samples, replace=True)
    ds = ds.drop(dataset.index)
    none_ds = ds[ds[target_type] == 0].sample(train_samples, replace=True)
    ds = ds.drop(none_ds.index)
    train_ds = pd.concat([dataset, none_ds]).sample(frac=1)
    train_ds.to_csv(train_csv_file, index=None, header=True)

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
cache_location = config['dataset']['cache_location']
train_base_image_path = config['dataset']['train_base_image_path']
test_base_image_path = config['dataset']['test_base_image_path']
use_cache = config['dataset']['use_cache']
output_test_images = config['dataset']['output_test_images']
random_train_image_transformation = config['dataset']['random_train_image_transformation']

train_csv_file, test_csv_file = create_specialized_csv(target_type, train_samples, test_samples, load_existing_cache, data_location)

batch_dataset_train = bds.BatchDataset(train_csv_file, batch_size)
data_generator_train = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_train,
    image_width,
    image_height,
    train_base_image_path,
    include_resized_mini_images=True,
    output_test_images=output_test_images,
    queue_workers=3,
    queue_size=50,
    color=True,
    random_image_transformation=random_train_image_transformation)
    #zip_path='./data/stage_1_train_images/rsna-intracranial-hemorrhage-detection.zip')

batch_dataset_test = bds.BatchDataset(test_csv_file, batch_size)
data_generator_test = Data_Generator.Data_Generator(
    target_type,
    batch_dataset_test,
    image_width,
    image_height,
    test_base_image_path,
    include_resized_mini_images=True,
    queue_workers=3,
    queue_size=50,
    color=True,
    random_image_transformation=False)
    #zip_path='./data/stage_1_train_images/rsna-intracranial-hemorrhage-detection.zip')

if use_cache:
    print("Setting cache")
    data_generator_cache = Data_Generator_Cache(
        cache_location,
        keep_existing_cache=load_existing_cache,
    )
    data_generator_train.set_data_generator_cache(data_generator_cache)
    data_generator_test.set_data_generator_cache(data_generator_cache)

data_generator_train.start_batcher()
data_generator_test.start_batcher()


# In[20]: create model
base_model_trainable = config['model']['base_model_trainable']

#with strategy.scope():
#base_model = tf.keras.applications.MobileNetV2(input_shape=(data_generator_train.image_height, data_generator_train.image_width, 1),
#                                               include_top=False,
#                                               weights='imagenet')

#base_model = tf.keras.applications.ResNet50()
base_model = tf.compat.v2.keras.applications.ResNet50(
    include_top=False,
    input_shape=(data_generator_train.image_height, data_generator_train.image_width, 3)
)
base_model.trainable = base_model_trainable

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

optimizer = Adam(lr=0.000100)
model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

print(model.summary())

# checkpoint
best_val_loss = sys.float_info.max
plotData = collections.defaultdict(list)
model_path = config['model']['model_path'] + 'model_{0}'.format(target_type)
print('#### Going to save model to: ' + model_path)
model.save(model_path)

load_existing_weights = config['model']['load_existing_weights']
if load_existing_weights:
    model.load_weights(model_path)

start = time.time()
plotData = collections.defaultdict(list)

# Tensorboard logging
tensorboar_log_dir = config['model']['tensorboar_log_dir']
log_dir=tensorboar_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch = 3, update_freq='epoch')

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

    history = model.fit_generator(
        generator=data_generator_train,
        steps_per_epoch=steps_per_epoch_size,
        epochs=1,
        verbose=1,
        use_multiprocessing=False,
        workers=0,
        max_queue_size=32,
        callbacks=[tensorboard_callback])

    if i > 0 and i % epochs_between_testing == 0:
       validation_loss = validate_model(model, data_generator_test)
       tf.summary.scalar('validation_loss', data=validation_loss, step=i)

    #train_accuracy = history.history['acc'][0]
    train_loss = history.history['loss'][0]
    
    if display_train_loss:
        tf.summary.scalar('train_loss', data=train_loss, step=i)
        plotData['train_loss'].append(train_loss)
    if display_validation_loss:
        plotData['validation_loss'].append(validation_loss)

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss
        model.save_weights(model_path)
        print("New best test loss!")
        # live_plot(plotData, logarithmic=True, averaging_size=1)
        #print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))

    if time.time() - start > 60:
        start = time.time()
        # live_plot(plotData, logarithmic=True, averaging_size=1)
        #print("AT:", round(train_accuracy, 5), " LT: ", round(train_loss, 5))


#%%
