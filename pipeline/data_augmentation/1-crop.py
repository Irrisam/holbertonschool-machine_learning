#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''

import tensorflow as tf


def crop_image(image, size):
    '''crops an image'''
    return (tf.image.random_crop(image, size))
