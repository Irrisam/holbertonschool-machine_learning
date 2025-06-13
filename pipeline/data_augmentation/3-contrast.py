#!/usr/bin/env python3
"""
contrasts an image
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    randomly adjusts the contrast
    """
    return tf.image.random_contrast(image, lower, upper)
