import tensorflow as tf
import numpy as np
#from augmentation.gaussian_filter import GaussianBlur

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image


def Get_Negative_Mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


# def gaussian_filter(v1, v2):
#     k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
#     gaussian_ope = GaussianBlur(kernel_size=k_size, min=0.1, max=2.0)
#     [v1, ] = tf.py_function(gaussian_ope, [v1], [tf.float32])
#     [v2, ] = tf.py_function(gaussian_ope, [v2], [tf.float32])
#     return v1, v2



class Augment:
    # Default augmentations are on, certain poor combinations will be rejected
    def __init__(self):
        self.BANNED_AUGMENTATIONS = [ [self._brightness, self._contrast],
                                 [self._rotation, self._brightness],
                                 [self._flip, self._blur],
                                 [self._zoom, self._blur],
                                 [self._shift, self._flip],
                                 [self._rotation, self._blur],
                                 [self._flip, self._gain],
                                 [self._brightness, self._gain],
                                 [self._blur, self._brightness],
                                ]

        self.augmentations = [ self._shift, self._zoom, self._rotation, self._flip,
                            self._brightness, self._contrast,self._gain,self._blur,self._speckle]
        self.scale = 0.5


    def augfunc(self, sample):
        # Randomly apply two transformations
        augs = np.random.choice(self.augmentations, 2, replace=False)

        while not self.check_augs(augs) :
            augs = np.random.choice(self.augmentations, 2, replace=False)

        sample = augs[1]( augs[0](sample) )

        return sample


    def check_augs(self, augs) :
        for banned in self.BANNED_AUGMENTATIONS :
            if augs[0] in banned and augs[1] in banned :
                return False
        return True

    # Image Transformation functions
    def _rotation(self, x, degrees=180):
        #return image.random_rotation(x, degrees)
        return tf.image.rot90(x, k=np.random.randint(4))

    def _shift(self, x, shift=0.1):
        #return image.random_shift(x, wrg=shift, hrg=shift)
        rx = np.random.randint(120*shift)
        ry = np.random.randint(120*shift)
        return tf.roll(x, shift=[ry, rx, 0], axis=[0, 1, 2])

    def _zoom(self, x, zrng=0.2):
        #return image.random_zoom(x, zoom_range=(1-zrng, 1+zrng))
        boxes = tf.random.uniform(shape=(1, 4))
        box_indices = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.int32)

        return tf.image.crop_and_resize( tf.expand_dims(x, axis=0), boxes, box_indices, (120,120), method="bilinear")

    def _flip(self, x):
        x = tf.image.random_flip_left_right(x)
        return tf.image.random_flip_up_down(x) 


    # Pixel Transformation functions
    def _brightness(self, x):
        return tf.image.random_brightness(x, max_delta=0.8 * self.scale)

    def _contrast(self, x):
        return tf.image.random_contrast(
            x, lower=1 - 0.8 * self.scale, upper=1 + 0.8 * self.scale
        )

    def _gain(self, x):
        g = np.random.uniform(-self.scale, self.scale)
        return tf.image.adjust_gamma(x, gamma=1.0, gain=g)

    def _speckle(self, x):
        prob_range=[0.0, 0.02]
        spec_value=5.

        prob = tf.random.uniform((), *prob_range)
        sample = tf.random.uniform(tf.shape(x))
        noisy_image = tf.where(sample <= prob, -spec_value*tf.ones_like(x), x)
        noisy_image = tf.where(sample >= (1. - prob), spec_value*tf.ones_like(x), noisy_image)
        return noisy_image

    def _blur(self, x):
        # SimClr implementation is applied at 10% of image size with a random sigma
        p = np.random.uniform(0.1, 2)
        if type(x) == np.ndarray:
            return cv2.GaussianBlur(x, (5, 5), p)
        return cv2.GaussianBlur(x.numpy(), (5, 5), p)
