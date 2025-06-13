#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''

import tensorflow as tf


def crop_image(image, size):
    return tf.image.resize_with_crop_or_pad(image, size[0], size[1])
