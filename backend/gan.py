from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np
import pickle

class GAN:
    
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = BinaryCrossentropy(from_logits=True)

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = GAN.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = GAN.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return GAN.cross_entropy(tf.ones_like(fake_output), fake_output)

    def __init__(self, discriminator, generator, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=0, beta_2=0.9):
        self.disc_input = discriminator.input_shape
        self.discriminator = discriminator
        self.generator = generator

        self.generator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def train(self, data, batch_size, n_batches, additional_d_steps):
        """[summary]

        Args:
            data ([type]): [description]
            batch_size ([type]): [description]
            n_batches ([type]): [description]
            additional_d_steps ([type]): [description]
        """ 
        progress = Progbar(n_batches)

        for n_batch in range(n_batches):

            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]))
            batch = data[batch_idx]

            self.train_step(batch, batch_size, additional_d_steps)

            self.train_hook(n_batch)

            progress.update(n_batch + 1)

    @tf.function
    def train_step(self, data, batch_size, additional_d_steps):

        for _ in range(additional_d_steps):
            noise = tf.random.normal([batch_size, self.disc_input[1], self.disc_input[2]*2 - 1, 3])
            generated_data = self.generator(noise, training=False)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(data, training=True)
                fake_output = self.discriminator(generated_data, training=True)
                disc_loss = GAN.discriminator_loss(real_output, fake_output)
            
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, self.disc_input[1], self.disc_input[2]*2 - 1, 3])
        generated_data = self.generator(noise, training=False)

        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            disc_loss = GAN.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        noise = tf.random.normal([batch_size, self.disc_input[1], self.disc_input[2]*2 - 1, 3])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=False)
            gen_loss = GAN.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def save(self, folder_path):
        self.generator.save(folder_path)

    def load(folder_path):
        generator = load_model(folder_path)
        
        return generator

    def train_hook(self, n_batch):
        pass