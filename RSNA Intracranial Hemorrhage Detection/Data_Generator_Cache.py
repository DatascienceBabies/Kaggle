import os
import tables
import numpy as np
from threading import Lock

class Data_Generator_Cache:
    _cache_location = None
    _hdf5_file = None
    _image_storage = None
    _key_storage = None
    _batch_dataset_mutex = Lock()
    _key_index_dictionary = {}
    _last_index_position = None

    def __init__(
        self,
        cache_location,
        width,
        height,
        key_length,
        keep_existing_cache=False):

        self._last_index_position = 0

        self._cache_location = cache_location
        data_shape = (0, height, width, 3)
        if keep_existing_cache:
            print('##### Warning, an existing data cache ' + self._cache_location + ' is being used #####')
            self._hdf5_file = tables.open_file(cache_location, mode='a')
            self._image_storage = self._hdf5_file.root.images
            self._key_storage = self._hdf5_file.root.key
        else:
            self._hdf5_file = tables.open_file(cache_location, mode='w')
            self._image_storage = self._hdf5_file.create_earray(self._hdf5_file.root, 'images', tables.Float16Atom(), shape=data_shape)
            self._key_storage = self._hdf5_file.create_earray(self._hdf5_file.root, 'key', tables.StringAtom(key_length), shape=(0, 1))        


    def add_to_cache(self, image, key):
        self._batch_dataset_mutex.acquire()
        try:
            self._image_storage.append(np.expand_dims(image,0))
            self._key_storage.append([[str.encode(key)]])
            self._key_index_dictionary[key] = self._last_index_position
            self._last_index_position = self._last_index_position + 1
        finally:
            self._batch_dataset_mutex.release()

    def key_exists(self, key):
        return key in self._key_index_dictionary

    def get_image(self, key):
        return self._image_storage[self._key_index_dictionary[key]]