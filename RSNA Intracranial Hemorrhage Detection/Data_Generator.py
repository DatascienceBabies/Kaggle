from skimage.io import imread
from skimage.transform import resize
import numpy as np
import BatchDataset as bds
from keras.utils import Sequence
import pydicom as dicom
import os
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import cv2
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import queue
from threading import Thread, Lock
import time
import math
import scipy.misc
import uuid

# This data generator is hardwired for Y being images
class Data_Generator(Sequence):
    batch_dataset = None
    one_hot_encoder = None
    archive = None
    archive_mutex = None
    base_image_path = ""
    image_width = None
    image_height = None
    batch_queue = None
    batch_dataset_mutex = None
    queue_workers = 6
    queue_size = 30

    def open_dcm_image(self, ID):
        try:
            picture_path = os.path.join(self.base_image_path, ID[0:12] + ".dcm")
            dcm_image = dicom.dcmread(picture_path)
            return self.get_dcm_image_data(dcm_image)
        except:
            print('Exception when fetching pixel array of file ', ID)
        
    def open_dcm_image_from_zip(self, ID):
        try:
            # Access pixel data in more intelligent way for uncompressed images (recommended)
            file_path_within_zip = "{0}/{1}".format(self.base_image_path, ID[0:12] + ".dcm")
            self.archive_mutex.acquire()
            file = self.archive.read(file_path_within_zip)
            self.archive_mutex.release()
            file_bytes = io.BytesIO(file)
            dcm_image = dicom.dcmread(file_bytes)
            return self.get_dcm_image_data(dcm_image)            
        except:
            print('Exception when fetching pixel array of file ', ID)   


    def get_dcm_image_data(self, dcm_image):
        image = dcm_image.pixel_array  # returns a NumPy array for uncompressed images

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Move to positive pixel values (not needed as long as we do not convert to Hounsfield units?)
        if image.min() < 0:
            image = image - image.min()        

        # Convert to 0-255 according to https://www.leadtools.com/sdk/medical/dicom-spec17#targetText=The%20minimum%20actual%20Pixel%20Sample,Value%20(0028%2C0107).
        max_dcm_pixel_value = math.pow(2, dcm_image.BitsStored)
        image = image / max_dcm_pixel_value * 255

        # Sanity check
        if image.max() > 255:
            raise AttributeError("Image max out of range")

        # Create three pixel channels
        image = np.stack((image,)*3, axis=-1)

        # Check for sanity stuff
        
        # TODO: Read up on this
        # Convert to Hounsfield units (HU)
        # intercept = dcm_image.RescaleIntercept
        # slope = dcm_image.RescaleSlope
        
        # if slope != 1:
        #    image = slope * image.astype(np.float64)
        #    image = image.astype(np.int16)
            
        # image += np.int16(intercept)

        if np.isnan(image).any():
            raise AttributeError("Image contains NaN")

        #cv2.imwrite('c:/temp/' + str(uuid.uuid4()) + '.jpg', image)
        
        # return np.expand_dims(np.array(image, dtype=np.int16), 2)
        return image

    def __init__(self, batch_dataset, image_width, image_height, base_image_path="", zip_path=None):
        self.batch_dataset = batch_dataset
        self.image_width = image_width
        self.image_height = image_height
        self.batch_queue = queue.Queue(self.queue_size)
        self.batch_dataset_mutex = Lock()
        self.archive_mutex = Lock()
        self.base_image_path = base_image_path
        
        if zip_path != None:
            self.archive = zipfile.ZipFile(zip_path, 'r')

        # TODO: The categories should be configurable
        # We need categories defined, as we cannot trust the auto-category with batch data
        self.one_hot_encoder = OneHotEncoder(categories=[[0, 1]], handle_unknown='ignore')

        for _ in range(self.queue_workers):
            worker = threading.Thread(target=self._worker_queue_data, args=())
            worker.start()

    def _worker_queue_data(self):
        while 1:
            # Fetch the next csv chunk
            self.batch_dataset_mutex.acquire()
            dataset_chunk = self.batch_dataset.get_next_chunk()
            self.batch_dataset_mutex.release()

            # Add batch of image data as Y
            # Open first image to get width x height. Assume all other images to be same resolution
            images_data = np.zeros((dataset_chunk.dataset.shape[0], self.image_width, self.image_height, 3))
            index = 0
            for row in dataset_chunk.dataset.iterrows():
                # TODO: Send in width and height as constructor parameters
                if self.archive != None:
                    image_data = self.open_dcm_image_from_zip(row[1]['ID'])
                else:
                    image_data = self.open_dcm_image(row[1]['ID'])

                if image_data.shape[0] != self.image_width or image_data.shape[1] != self.image_height:
                    image_data = cv2.resize(image_data, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC).reshape(self.image_width, self.image_height, image_data.shape[2])

                images_data[index, :, :, :] = image_data
                index = index + 1

            X = images_data
            # TODO: The output(s) should be definable, but for now just create a one hot out of the binary epidural
            # Y = self.one_hot_encoder.fit_transform(dataset_chunk.dataset['epidural'].values.reshape(-1, 1))

            # TODO: Having some problems, testing out a simpler Y
            Y = dataset_chunk.dataset['epidural'].values

            self.batch_queue.put((X, Y))

    def __len__(self):
        return self.batch_dataset.batch_amount()

    def __getitem__(self, idx):
        try:
            item = self.batch_queue.get_nowait()
        except:
            # print("Unable to immediately fetch new databatch")
            item = self.batch_queue.get(block=True)
        X = item[0]
        Y = item[1]
        return X, Y