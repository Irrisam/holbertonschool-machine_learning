#!/usr/bin/env python3
'''
flips an image usig tensorflow
'''

import tensorflow as tf
import matplotlib.pyplot as plt


def rotate_image(image):
    '''crops an image'''
    return (tf.image.rot90(image, k=1, name=None))


def main():
    # Load the image
    image_path = "Original_Doge_meme.jpg"
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)

    # Rotate the image
    rotated_image = rotate_image(image)

    # Display the rotated image
    plt.figure(figsize=(8, 8))
    plt.imshow(rotated_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
