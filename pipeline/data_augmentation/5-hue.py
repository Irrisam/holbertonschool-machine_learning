#!/usr/bin/env python3
"""
contrasts an image
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    randomly adjusts the contrast
    """
    return tf.image.adjust_hue(image, delta)
