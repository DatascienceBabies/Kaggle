import os
#import tables
import h5py
import numpy as np
from threading import Lock
import time
import scipy

class Data_Generator_Cache:
    _cache_location = None
    _batch_dataset_mutex = None
    _key_index_dictionary = None
    _last_index_position = None
    _cache_mutex = None
    _color = None
    _channels = None
    _cache = None
    _last_flush_time = None

    def __init__(
        self,
        cache_location,
        width,
        height,
        key_length,
        keep_existing_cache=False,
        image_datatype=np.float32,
        start_size=30,
        color=True):

        self._last_index_position = 0
        self._cache_mutex = Lock()
        self._key_index_dictionary = {}
        self._batch_dataset_mutex = Lock()
        self._color = color
        self._last_flush_time = time.time()

        self._channels = 3
        if not self._color:
            self._channels = 1

        self._cache_location = cache_location
        if keep_existing_cache and os.path.exists(self._cache_location) and os.path.isfile(self._cache_location):
            print('##### Warning, an existing data cache ' + self._cache_location + ' is being used #####')
            # Check the contents of the file and cache the key - index
            print('Initialization data cache ' + self._cache_location)
            start_time = time.time()
            self._cache = h5py.File(cache_location, 'a')
            total_length = self._cache['key'].shape[0]
            items_left_to_cache = total_length
            self._last_index_position = 0
            while items_left_to_cache > 1000:
                batch_keys = self._cache['key'][self._last_index_position : self._last_index_position + 1000]
                for i in range(1000):
                    self._key_index_dictionary[batch_keys[i][0]] = self._last_index_position
                    self._last_index_position = self._last_index_position + 1
                items_left_to_cache = items_left_to_cache - 1000
                if time.time() - start_time >= 1:
                    start_time = time.time()
                    print ('\r', round(self._last_index_position/total_length*100, 1),' percent complete         ', end='')

            if items_left_to_cache > 0:
                batch_keys = self._cache['key'][self._last_index_position : self._last_index_position + items_left_to_cache]
                for i in range(items_left_to_cache):
                    self._key_index_dictionary[batch_keys[i][0]] = self._last_index_position
                    self._last_index_position = self._last_index_position + 1                
                items_left_to_cache = 0
            print ('\r', round(self._last_index_position/total_length*100, 1),' percent complete         ', end='')
            print('')
            
        else:
            self._cache = h5py.File(cache_location, 'w')
            self._cache.create_dataset("images", shape=(start_size, height, width, self._channels), maxshape=(None, height, width, self._channels), dtype=image_datatype, compression="gzip", shuffle=True)
            self._cache.create_dataset("key", shape=(start_size, 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))
            self._cache.create_dataset("current_location", shape=(1, 1), maxshape=(1, 1), dtype=np.int32)
            self._cache['current_location'][0][0] = 0


    def add_to_cache(self, image, key):
        try:
            self._cache_mutex.acquire()
            if self._last_index_position == self._cache['images'].shape[0]:
                self._cache['images'].resize((self._cache['images'].shape[0] * 2, self._cache['images'].shape[1], self._cache['images'].shape[2], self._cache['images'].shape[3]))
                self._cache['key'].resize((self._cache['key'].shape[0] * 2, self._cache['key'].shape[1]))

            self._cache['images'][self._last_index_position] = image
            self._cache['key'][self._last_index_position] = key
            self._cache['current_location'][0][0] = self._cache['current_location'][0][0] + 1
            self._key_index_dictionary[key] = self._last_index_position
            self._last_index_position = self._last_index_position + 1

            if time.time() - self._last_flush_time >= 60:
                self._last_flush_time = time.time()
                self._cache.flush()
        finally:
            self._cache_mutex.release()

    def key_exists(self, key):
        return key in self._key_index_dictionary

    def get_image(self, key):
        try:
            self._cache_mutex.acquire()

            if time.time() - self._last_flush_time >= 60:
                self._last_flush_time = time.time()
                self._cache.flush()

            return self._cache['images'][self._key_index_dictionary[key]]
        finally:
            self._cache_mutex.release()

    def get_id_of_last_fetched_image(self):
        for id, index in self._key_index_dictionary.items():
            if index == self._cache['current_location'][0][0]:
                return id
        raise ValueError('Could not identify last fetched image in data generator cache')