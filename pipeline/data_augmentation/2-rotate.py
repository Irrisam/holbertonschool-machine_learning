#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''

import tensorflow as tf


def rotate_image(image):
    '''crops an image'''
    return (tf.image.rot90(image, k=1, name=None))
