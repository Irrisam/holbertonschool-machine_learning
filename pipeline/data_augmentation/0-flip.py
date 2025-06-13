#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''
import tensorflow as tf


def flip_image(image):
    """flips the image

    Args:
        image (image): the image
    """
    return tf.image.flip_left_right(image)
