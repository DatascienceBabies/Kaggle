from skimage.io import imread
from skimage.transform import resize
import numpy as np
import BatchDataset as bds
from tensorflow.python.keras.utils import Sequence
import pydicom as dicom
import os
from tensorflow.python.keras.utils import to_categorical
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
import skimage as sk
import time
import random
from Data_Generator_Cache import Data_Generator_Cache
import Image_Transformer

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
    queue_workers = None
    queue_size = None
    output_test_images = None
    last_test_image_created = None
    include_resized_mini_images = None
    original_width = None
    original_height = None
    cache_data = None
    data_generator_cache = None
    target_type = None
    random_image_transformation = None
    image_transformer = None
    

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
        # if image.min() < 0:
        #     image = image - image.min()        

        # TODO: Really check how the pixel resizing should be done here...
        # Convert to 0-255 according to https://www.leadtools.com/sdk/medical/dicom-spec17#targetText=The%20minimum%20actual%20Pixel%20Sample,Value%20(0028%2C0107).
        # max_dcm_pixel_value = math.pow(2, dcm_image.BitsStored)
        # image = image / max_dcm_pixel_value * 255
        # image = image / image.max() * 255

        # Sanity check
        # if image.max() > 255:
        #     raise AttributeError("Image max out of range")

        # Create three pixel channels
        if self.color:
            image = np.stack((image,)*3, axis=-1)
        else:
            image = np.stack((image,)*1, axis=-1)

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
        
        # return np.expand_dims(np.array(image, dtype=np.int16), 2)
        return image

    def __init__(
        self,
        target_type,
        batch_dataset,
        image_width,
        image_height,
        base_image_path="",
        zip_path=None,
        output_test_images=False,
        include_resized_mini_images=False,
        cache_data=False,
        cache_location=None,
        keep_existing_cache=False,
        queue_workers=1,
        queue_size=30,
        color=True,
        random_image_transformation=False):
        
        self.queue_size = queue_size
        self.batch_dataset = batch_dataset
        self.image_width = image_width
        self.image_height = image_height
        self.batch_queue = queue.Queue(self.queue_size)
        self.batch_dataset_mutex = Lock()
        self.archive_mutex = Lock()
        self.base_image_path = base_image_path
        self.output_test_images = output_test_images
        self.last_test_image_created = time.time()
        self.include_resized_mini_images = include_resized_mini_images
        self.original_width = image_width
        self.original_height = image_height
        self.cache_data = cache_data
        self.target_type = target_type
        self.queue_workers = queue_workers
        self.color = color
        self.random_image_transformation = random_image_transformation

        if (self.random_image_transformation):
            self.image_transformer = Image_Transformer.Image_Transformer()

        if (self.include_resized_mini_images):
            self.image_width = math.ceil(self.image_width * 1.5)
        
        if zip_path != None:
            self.archive = zipfile.ZipFile(zip_path, 'r')

        # TODO: The categories should be configurable
        # We need categories defined, as we cannot trust the auto-category with batch data
        self.one_hot_encoder = OneHotEncoder(categories=[[0, 1]], handle_unknown='ignore')

        if self.cache_data:
            self.data_generator_cache = Data_Generator_Cache(
                cache_location,
                self.image_width,
                self.image_height,
                keep_existing_cache=keep_existing_cache,
                key_length=12,
                color=self.color
            )

        for _ in range(self.queue_workers):
            worker = threading.Thread(target=self._worker_queue_data, args=())
            worker.start()

    def _resize_image(self, image, width, height):
        width = int(width)
        height = int(height)
        return cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC).reshape(height, width, image.shape[2])

    def _add_smaller_images(self, image):
        new_width = math.floor(image.shape[1] / 2)
        new_height = math.floor(image.shape[0] / 2)
        x_offset=image.shape[1]
        resized_images = []

        for i in range(4):
            resized_images.append(self._resize_image(image, new_width, new_height))
            new_width = math.floor(new_width / 2)
            new_height = math.floor(new_height / 2)
        
        image = cv2.copyMakeBorder(image, 0, 0, 0, math.floor(image.shape[1] / 2), cv2.BORDER_CONSTANT)
        if not self.color:
            image = np.stack((image,)*1, axis=-1)
        y_offset=0
        for i in range(4):
            resized_image = resized_images[i]
            image[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image
            y_offset = y_offset + resized_image.shape[0]

        return image                   


    def _worker_queue_data(self):
        while 1:
            try:
                # Fetch the next csv chunk
                self.batch_dataset_mutex.acquire()
                dataset_chunk = self.batch_dataset.get_next_chunk()
                self.batch_dataset_mutex.release()

                # Add batch of image data as Y
                # Open first image to get width x height. Assume all other images to be same resolution
                if self.color:
                    images_data = np.zeros((dataset_chunk.dataset.shape[0], self.image_height, self.image_width, 3))
                else:
                    images_data = np.zeros((dataset_chunk.dataset.shape[0], self.image_height, self.image_width, 1))
                index = 0
                for row in dataset_chunk.dataset.iterrows():
                    image = None
                    if self.cache_data and self.data_generator_cache.key_exists(row[1]['ID']):
                        image = self.data_generator_cache.get_image(row[1]['ID'])
                    else:
                        # TODO: Send in width and height as constructor parameters
                        if self.archive != None:
                            image = self.open_dcm_image_from_zip(row[1]['ID'])
                        else:
                            image = self.open_dcm_image(row[1]['ID'])

                        if image.shape[1] != self.original_width or image.shape[0] != self.original_height:
                            image = self._resize_image(image, self.original_width, self.original_height)

                        if self.include_resized_mini_images:
                            image = self._add_smaller_images(image)

                        if self.cache_data:
                            self.data_generator_cache.add_to_cache(image, row[1]['ID'])

                    if self.random_image_transformation:
                        image = self.image_transformer.random_transforms(image)

                    images_data[index, :, :, :] = image
                    index = index + 1

                X = images_data
                Y = np.eye(2)[dataset_chunk.dataset[self.target_type].values]

                self.batch_queue.put((X, Y))
            except Exception as e:
                print('Failed to queue image with ID ' + row[1]['ID'] + ' with exception ' + str(e))
                os._exit(1)

    def _store_debug_picture(self, X, Y):
        if time.time() - self.last_test_image_created > 10:
            # Save one random picture
            self.last_test_image_created = time.time()
            index = random.randint(0, X.shape[0]-1)
            cv2.imwrite('c:/temp/' + str(uuid.uuid4()) + ' - ' + str(Y[index]) + '.jpg', X[index])

    def __len__(self):
        return self.batch_dataset.batch_amount()

    def __getitem__(self, idx):
        try:
            item = self.batch_queue.get_nowait()
        except:
            print("Unable to immediately fetch new databatch")
            item = self.batch_queue.get(block=True)
        X = item[0]
        Y = item[1]

        if self.output_test_images:
            self._store_debug_picture(X, Y)

        return X.astype(int), Y.astype(int)

    # TODO: Well, this is stupid. But I have to try it out...
    current_item = None
    def getitem_tensorflow_2_X(self):
        try:
            self.current_item = self.batch_queue.get_nowait()
        except:
            print("Unable to immediately fetch new databatch")
            self.current_item = self.batch_queue.get(block=True)
        X = self.current_item[0]

        if self.output_test_images:
            Y = self.current_item[1]
            self._store_debug_picture(X, Y)

        return X.astype(int)

    def getitem_tensorflow_2_Y(self):
        Y = self.current_item[1]

        if self.output_test_images:
            X = self.current_item[0]
            self._store_debug_picture(X, Y)

        return Y.astype(int)