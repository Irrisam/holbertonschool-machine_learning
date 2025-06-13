#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''

import tensorflow as tf


def crop_image(image, size):
    return (tf.image.crop_and_resize(
        image,
        boxes=1,
        box_indices=[0, 1],
        crop_size=size
    )
    )
