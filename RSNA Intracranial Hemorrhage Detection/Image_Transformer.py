import cv2
import random
import numpy as np

class Image_Transformer():

    def _rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result.reshape(image.shape)

    def _flip(self, image, type):
        return cv2.flip(image, type).reshape(image.shape)

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
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == 3:
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy.reshape(image.shape)
    
    def random_transforms(self, image):
        if random.uniform(0, 1) > 0.3:
            image = self._rotate_image(image, random.uniform(-15, +15))
        if random.uniform(0, 1) > 0.3:
            image = self._flip(image, random.randint(-1, 1))
        #if random.uniform(0, 1) > 0.3:
            #image = self._noise(image, random.randint(0, 3))

        return image