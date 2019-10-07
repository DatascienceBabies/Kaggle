import os
#import tables
import h5py
import numpy as np
from threading import Lock
import time

class Data_Generator_Cache:
    _cache_location = None
    _batch_dataset_mutex = None
    _key_index_dictionary = None
    _last_index_position = None
    _cache_mutex = None
    _color = None
    _channels = None

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

        self._channels = 3
        if not self._color:
            self._channels = 1

        self._cache_location = cache_location
        if keep_existing_cache and os.path.exists(self._cache_location) and os.path.isfile(self._cache_location):
            print('##### Warning, an existing data cache ' + self._cache_location + ' is being used #####')
            # Check the contents of the file and cache the key - index
            print('Initialization data cache ' + self._cache_location)
            start_time = time.time()
            with h5py.File(cache_location, 'r') as file:
                total_length = file['key'].shape[0]
                for key in file['key']:
                    if time.time() - start_time >= 1:
                        start_time = time.time()
                        print ('\r', round(self._last_index_position/total_length*100, 1),' percent complete         ', end='')
                    self._key_index_dictionary[key[0]] = self._last_index_position
                    self._last_index_position = self._last_index_position + 1
                print('')
            
        else:
            with h5py.File(cache_location, 'w') as file:
                file.create_dataset("images", shape=(start_size, height, width, self._channels), maxshape=(None, height, width, self._channels), dtype=image_datatype)
                file.create_dataset("key", shape=(start_size, 1), maxshape=(None, 1), dtype=h5py.special_dtype(vlen=str))


    def add_to_cache(self, image, key):
        try:
            self._cache_mutex.acquire()
            with h5py.File(self._cache_location, 'a') as file:
                if self._last_index_position == file['images'].shape[0]:
                    file['images'].resize((file['images'].shape[0] * 2, file['images'].shape[1], file['images'].shape[2], file['images'].shape[3]))
                    file['key'].resize((file['key'].shape[0] * 2, file['key'].shape[1]))

                file['images'][self._last_index_position] = image
                file['key'][self._last_index_position] = key
                self._key_index_dictionary[key] = self._last_index_position
                self._last_index_position = self._last_index_position + 1
        finally:
            self._cache_mutex.release()

    def key_exists(self, key):
        return key in self._key_index_dictionary

    def get_image(self, key):
        try:
            self._cache_mutex.acquire()
            with h5py.File(self._cache_location, 'r') as file:
                return file['images'][self._key_index_dictionary[key]]
        finally:
            self._cache_mutex.release()