import cv2
import random
import numpy as np
import logging
import math

class Image_Transformer():

    def __init__(self):
        logging.basicConfig(filename='loggOutput.log',level=logging.DEBUG)

    def _rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result.reshape(image.shape)

    def _flip(self, image, type):
        return cv2.flip(image, type).reshape(image.shape)

    def _stretch(self, image, width_ratio=0.1, height_ratio=0):
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.copyMakeBorder(
            image,
            math.floor(height * height_ratio / 2) + 1,
            math.floor(height * height_ratio / 2) + 1,
            math.floor(width * width_ratio / 2) + 1,
            math.floor(width * width_ratio / 2) + 1,
            cv2.BORDER_CONSTANT)           

        #image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC).reshape(height, width, image.shape[2])
        image = cv2.resize(image, dsize=(width, height))

        return image

    def _noise(self, image, noise_typ):
        if noise_typ == 0:
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy.reshape(image.shape)
        elif noise_typ == 1:
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out.reshape(image.shape)
        elif noise_typ == 2:
            # TODO: Fix issue
            #vals = len(np.unique(image))
            #vals = 2 ** np.ceil(np.log2(vals))
            #noisy = np.random.poisson(image * vals) / float(vals)
            #return noisy
            return image
        elif noise_typ == 3:
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy.reshape(image.shape)
    
    def random_transforms(self, image):
        image_safe = np.copy(image)
        try:
            if random.uniform(0, 1) > 0.3:
                image = self._rotate_image(image, random.uniform(-20, +20))
            if random.uniform(0, 1) > 0.3:
                image = self._flip(image, random.randint(-1, 1))
            if random.uniform(0, 1) > 0.3:
                image = self._noise(image, random.randint(0, 3))
            if random.uniform(0, 1) > 1.0:
                image = self._stretch(image, width_ratio=random.uniform(0, 0.2), height_ratio=random.uniform(0, 0.2))
        except Exception as e:
            print('Image_Transformer failed with exception: ' + str(e))
            logging.error('Image_Transformer failed with exception: ' + str(e))
            image = image_safe

        return image