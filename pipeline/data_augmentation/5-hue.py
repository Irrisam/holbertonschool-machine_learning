#!/usr/bin/env python3
"""
contrasts an image
"""

import tensorflow as tf
import matplotlib.pyplot as plt


def change_hue(image, delta):
    """
    randomly adjusts the contrast
    """
    return tf.image.adjust_hue(image, delta)


def main():
    # Load the image
    image_path = "Original_Doge_meme.jpg"
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)

    # Rotate the image
    rotated_image = change_hue(image, -0.5)

    # Display the rotated image
    plt.figure(figsize=(8, 8))
    plt.imshow(rotated_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
