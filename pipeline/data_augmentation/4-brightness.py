#!/usr/bin/env python3
"""
contrasts an image
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    randomly adjusts the contrast
    """
    return tf.image.random_brightness(image, max_delta)
