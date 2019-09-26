import os
import tables
import numpy as np

class Data_Generator_Cache:
    _cache_location = None
    _hdf5_file = None
    _image_storage = None
    _key_storage = None
    _byte_array = None

    def __init__(
        self,
        cache_location,
        width,
        height,
        key_length):

        self._cache_location = cache_location
        data_shape = (0, height, width, 3)
        self._hdf5_file = tables.open_file(cache_location, mode='w')
        self._image_storage = self._hdf5_file.create_earray(self._hdf5_file.root, 'images', tables.Float16Atom(), shape=data_shape)
        self._key_storage = self._hdf5_file.create_earray(self._hdf5_file.root, 'key', tables.StringAtom(key_length), shape=(0, 1))
        self._byte_array = bytearray()
        


    def add_to_cache(self, image, key):
        self._image_storage.append(np.expand_dims(image,0))
        self._key_storage.append([[str.encode(key)]])

    def key_exists(self, key):
        return len(np.where(self._key_storage[:] == str.encode(key))[0]) > 0

    def get_image(self, key):
        return self._image_storage[np.where(self._key_storage[:] == str.encode(key))[0][0]]