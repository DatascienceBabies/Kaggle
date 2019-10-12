import os
#import tables
import h5py
import numpy as np
from threading import Lock
import time
import scipy
import glob
import logging

class Data_Generator_Cache:
    _cache_location = None
    _keep_existing_cache = None
    _image_ram_cache = None

    def __init__(
        self,
        cache_location,
        keep_existing_cache=False):

        self._cache_location = cache_location
        self._keep_existing_cache = keep_existing_cache
        self._image_ram_cache = {}

        if self._keep_existing_cache == False:
            fileList = glob.glob(self._cache_location + '/*.npz')
            for filePath in fileList:
                try:
                    os.remove(filePath)
                except:
                    print("Error while deleting file : ", filePath)

    def add_to_cache(self, image, key):
        path = os.path.join(self._cache_location, key)
        np.savez_compressed(path, a=image)

    def key_exists(self, key):
        try:
            path = os.path.join(self._cache_location, key + '.npz')
            if os.path.exists(path) == False:
                return False
            path = os.path.join(self._cache_location, key + '.npz')
            image = np.load(path)['a']
            self._image_ram_cache[key] = image
        except:
            return False

    def get_image(self, key):
        return self._image_ram_cache.pop(key)