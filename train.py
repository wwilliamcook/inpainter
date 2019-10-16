"""Trains an image inpainting CNN.

Uses the high res DIV2K dataset (https://data.vision.ee.ethz.ch/cvl/DIV2K/).

References:
* https://www.tensorflow.org/tutorials/load_data/images
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset
* https://www.tensorflow.org/tutorials/keras/save_and_load
"""

import tensorflow as tf
import numpy as np
import pathlib
import os
import zipfile

from models import generate, save
from train_utils import train
from mask_generator import buildMaskGenerator


BATCH_SIZE = 16
EPOCHS = 1
STEPS_PER_EPOCH = 200
IMG_HEIGHT = 720
IMG_WIDTH = 640
TRAIN_DATA_DIR = './data/DIV2K_train_HR'


AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = pathlib.Path(TRAIN_DATA_DIR).absolute()


train_image_count = len(list((data_dir).glob('*.png')))
print('Loading {} training images.'.format(train_image_count))

img_list_ds = tf.data.Dataset.list_files(str(data_dir/'*.png'))
# Shuffle and repeat forever
img_list_ds = img_list_ds.shuffle(buffer_size=1000).repeat()

def random_rescale(image):
    """Randomly rescales an image.
    """
    min_size = tf.convert_to_tensor([IMG_HEIGHT, IMG_WIDTH], tf.float32)
    shape = tf.cast(tf.shape(image), tf.float32)[:2]
    min_scale = tf.reduce_max(min_size / shape)
    scale = tf.random.uniform([], minval=min_scale, maxval=1)
    new_shape = tf.cast(tf.round(shape * scale), tf.int32)
    return tf.image.resize(image, new_shape)

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Convert to floats in the [0,1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Map to [-1, 1]
    img = img * 2. - 1.
    # Randomly scale the image
    random_rescale(img)
    # Randomly crop to desired size
    img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, 3])
    # Randomly flip horizontally
    img = tf.image.random_flip_left_right(img)
    return img

def process_path(img_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img

# Make a dataset of images
img_ds = img_list_ds.map(process_path,
                         num_parallel_calls=AUTOTUNE)
# Wrap a mask generator to make a dataset of masks
mask_ds = tf.data.Dataset.from_generator(
    buildMaskGenerator(IMG_WIDTH, IMG_HEIGHT),
    tf.float32)
# Zip the two datasets together
img_mask_ds = tf.data.Dataset.zip((img_ds, mask_ds))
# Batch
train_ds = img_mask_ds.batch(BATCH_SIZE)
# Let the dataset fetch batches in the background
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

print('Successfully loaded training dataset.')

print('Training.')
train(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

print('Saving models.')
save()
print('Models saved.')
