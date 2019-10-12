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
    _ctr_images_saved = None
    _ctr_images_fetched = None
    _time_since_last_info = None
    _output_debug_information = None

    def __init__(
        self,
        cache_location,
        keep_existing_cache=False,
        output_debug_information=True):

        self._cache_location = cache_location
        self._keep_existing_cache = keep_existing_cache
        self._time_since_last_info = time.time()
        self._output_debug_information = output_debug_information
        self._ctr_images_saved = 0
        self._ctr_images_fetched = 0

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
        self._ctr_images_saved = self._ctr_images_saved + 1

    def get_image(self, key):
        try:
            path = os.path.join(self._cache_location, key + '.npz')
            if os.path.exists(path) == False:
                return None
            path = os.path.join(self._cache_location, key + '.npz')
            image = np.load(path)['a']
            self._ctr_images_fetched = self._ctr_images_fetched + 1

            if self._output_debug_information and time.time() - self._time_since_last_info > 2:
                print('Fetched ratio: ' + str(round(self._ctr_images_fetched / (self._ctr_images_fetched + self._ctr_images_saved), 2)) + '. Images saved: ' + str(self._ctr_images_saved) + ". Images fetched from cache: " + str(self._ctr_images_fetched))
                self._time_since_last_info = time.time()
            return image
        except:
            return None