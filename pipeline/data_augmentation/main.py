import tensorflow as tf
import matplotlib.pyplot as plt

image_path = "/Users/tristan/Desktop/HS/holbertonschool-machine_learning/pipeline/data_augmentation/Original_Doge_meme.jpg"
image = tf.io.read_file(image_path)
image = tf.image.decode_png(image, channels=3)  # or channels=4 for RGBA
image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
image = tf.expand_dims(image, 0)  # add batch dimension


def crop_image(image, boxes, size):
    '''
    crops an image
    '''
    batch_size = tf.shape(image)[0]
    box_indices = tf.range(tf.shape(boxes)[0])

    return tf.image.crop_and_resize(
        image,
        boxes=boxes,
        box_indices=box_indices,
        crop_size=size
    )


# Define crop boxes (normalized coordinates: [y_min, x_min, y_max, x_max])
boxes = tf.constant([
    [0.1, 0.1, 0.5, 0.5],  # top-left crop
    [0.5, 0.5, 0.9, 0.9],  # bottom-right crop
])

# Test the function
cropped = crop_image(image, boxes, size=[224, 224])

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image[0])
axes[0].set_title('Original')
axes[0].axis('off')

# Cropped images
for i in range(2):
    axes[i+1].imshow(cropped[i])
    axes[i+1].set_title(f'Crop {i+1}')
    axes[i+1].axis('off')

plt.show()
