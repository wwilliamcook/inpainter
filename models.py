import tensorflow as tf
import tensorflow.keras.layers as kl
import pathlib
import time


generator = tf.keras.models.Sequential([
    kl.Conv2D(32, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(32, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(64, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(64, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(128, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(128, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(256, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(256, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(128, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(128, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(64, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(64, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(32, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(32, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(3, [5, 5], padding='SAME'),
    kl.BatchNormalization(),
    kl.Activation('tanh')
    ])

discriminator = tf.keras.models.Sequential([
    kl.Conv2D(32, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(64, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(128, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.Conv2D(256, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.MaxPooling2D(),
    kl.Conv2D(512, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.MaxPooling2D(),
    kl.Conv2D(1024, [5, 5]),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dropout(.3),
    kl.GlobalMaxPooling2D(),
    kl.Dense(512),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dense(256),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dense(128),
    kl.BatchNormalization(),
    kl.Activation('relu'),
    kl.Dense(1)
    ])

@tf.function
def generate(image_batch, mask_batch):
    masked_batch = image_batch * mask_batch
    concat_batch = tf.concat([masked_batch, mask_batch], axis=3)
    generated = generator(concat_batch)
    merged = tf.where(mask_batch >= .5, image_batch, generated)
    return merged

def save(model_dir='./trained_models'):
    """Save the models."""
    model_dir = pathlib.Path(model_dir)
    time_str = time.asctime().replace(' ', '-')
    generator.save(str(model_dir/'generator_{}.h5'.format(time_str)))
    discriminator.save(str(model_dir/'discriminator_{}.h5'.format(time_str)))
    return str(model_dir)
