from skimage.io import imread
from skimage.transform import resize
import numpy as np
import BatchDataset as bds
from keras.utils import Sequence
import pydicom as dicom
import os
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# This data generator is hardwired for Y being images
class Data_Generator(Sequence):
    batch_dataset = None
    one_hot_encoder = None

    def open_dcm_image(self, picture_folder, ID):
        picture_path = os.path.join(picture_folder, ID[0:12] + ".dcm")
        dcm_image = dicom.dcmread(picture_path)
        
        # Access pixel data in more intelligent way for uncompressed images (recommended)
        image = dcm_image.pixel_array  # returns a NumPy array for uncompressed images

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 1
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # TODO: Read up on this
        # Convert to Hounsfield units (HU)
        # intercept = dcm_image.RescaleIntercept
        # slope = dcm_image.RescaleSlope
        
        # if slope != 1:
        #    image = slope * image.astype(np.float64)
        #    image = image.astype(np.int16)
            
        # image += np.int16(intercept)

        # Move to positive pixel values (not needed as long as we do not convert to Hounsfield units?)
        if image.min() < 0:
            image = image - image.min()
        
        return np.expand_dims(np.array(image, dtype=np.int16), 2)

    def __init__(self, batch_dataset):
        self.batch_dataset = batch_dataset

        # TODO: The categories should be configurable
        # We need categories defined, as we cannot trust the auto-category with batch data
        self.one_hot_encoder = OneHotEncoder(categories=[[0, 1]], handle_unknown='ignore')  

    def __len__(self):
        return self.batch_dataset.batch_amount()

    def __getitem__(self, idx):
        # Fetch the next csv chunk
        dataset_chunk = self.batch_dataset.get_next_chunk()

        # Add batch of image data as Y
        # Open first image to get width x height. Assume all other images to be same resolution
        images_data = []
        index = 0
        for row in dataset_chunk.dataset.iterrows():
            image_data = self.open_dcm_image('./data/stage_1_train_images/', row[1]['ID'])
            if images_data == []:
                images_data = np.zeros((dataset_chunk.dataset.shape[0], image_data.shape[0], image_data.shape[1], 1))

            images_data[index, :, :, :] = self.open_dcm_image('./data/stage_1_train_images/', row[1]['ID'])
            index = index + 1

        X = images_data
        # TODO: The output(s) should be definable, but for now just create a one hot out of the binary any
        #Y = self.one_hot_encoder.fit_transform(dataset_chunk.dataset['any'].values.reshape(-1, 1))

        # TODO: Having some problems, testing out a simpler Y
        Y = dataset_chunk.dataset['any'].values

        return X, Y