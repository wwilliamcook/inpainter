"""Training utilities for an image inpainting CNN.

References:
* https://www.tensorflow.org/beta/tutorials/generative/dcgan
"""

import tensorflow as tf
import os
import time
import numpy as np

from models import generator, discriminator, generate


# Build loss functions
crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(discr_outputs):
    loss = crossentropy(tf.ones_like(discr_outputs), discr_outputs)
    return loss
def discriminator_loss(real_outputs, fake_outputs):
    real_loss = crossentropy(tf.ones_like(real_outputs), real_outputs)
    fake_loss = crossentropy(tf.zeros_like(fake_outputs), fake_outputs)
    return real_loss + fake_loss

# Create optimizers
generator_optimizer = tf.keras.optimizers.Adam(0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)

# Create checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

@tf.function
def train_step(image_batch, mask_batch):
    # Hide masked regions and concat masks onto images along channel axis
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # Patch masked images using generator
        patched_images = generate(image_batch, mask_batch)
        # Get discriminator outputs
        fake_d_out = discriminator(patched_images)
        real_d_out = discriminator(image_batch)
        # Get generator loss
        g_loss = generator_loss(fake_d_out)
        # Get discriminator loss
        d_loss = discriminator_loss(real_d_out, fake_d_out)
        # Get gradients
        g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
        d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
        # Apply gradients
        generator_optimizer.apply_gradients(
            zip(g_grad, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(d_grad, discriminator.trainable_variables))

def train(dataset, epochs=1, steps_per_epoch=None, checkpt_freq=None):
    step_len = int(np.ceil(np.log10(steps_per_epoch)))
    progbar_len = 30
    progbar_clear_str = '\b' * (2*step_len + 4 + progbar_len)
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch + 1))
        start = time.time()
        
        print('{}/{}\t[{}]'.format(
            '1'.rjust(step_len),
            steps_per_epoch,
            '.' * progbar_len), end='', flush=True)

        for i, (image_batch, mask_batch) in enumerate(dataset):
            progbar_fill = int(float(i) / float(steps_per_epoch) * progbar_len)
            if progbar_fill == 0:
                progbar_str = '.' * progbar_len
            elif progbar_fill == 1:
                progbar_str = '>' + '.' * (progbar_len - 1)
            else:
                progbar_str = '=' * (progbar_fill - 1) + '>' + '.' * (progbar_len - progbar_fill)
            print(progbar_clear_str + '{}/{}\t[{}]'.format(
                str(i + 1).rjust(step_len),
                steps_per_epoch,
                progbar_str), end='', flush=True)
            
            if i >= steps_per_epoch:
                break
            train_step(image_batch, mask_batch)
        
        print(progbar_clear_str + '{}/{}\t[{}]'.format(
            steps_per_epoch,
            steps_per_epoch,
            '=' * progbar_len))

        if checkpt_freq is not None:
            if (epoch + 1) % checkpt_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))

def test_ds(batch_size, im_w=32, im_h=32):
    while True:
        yield (np.float32(np.random.random(size=[batch_size, im_h, im_w, 3])),
               np.float32(np.random.randint(0, 2, size=[batch_size, im_h, im_w, 1])))

if __name__ == '__main__':
    train(test_ds(2, 256, 256), 5, 10)
