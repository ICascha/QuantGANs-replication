from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor
from math import floor, ceil

import pickle
import tensorflow as tf
import numpy as np

class GAN:
    """ Generative adverserial network class.

    Training code for a standard DCGAN using the Adam optimizer.
    Code taken in part from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
    """    

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    def __init__(self, discriminator, generator, training_input, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, from_logits=True):
        """Create a GAN instance

        Args:
            discriminator (tensorflow.keras.models.Model): Discriminator model.
            generator (tensorflow.keras.models.Model): Generator model.
            training_input (int): input size of temporal axis of noise samples.
            lr_d (float, optional): Learning rate of discriminator. Defaults to 1e-4.
            lr_g (float, optional): Learning rate of generator. Defaults to 3e-4.
            epsilon (float, optional): Epsilon paramater of Adam. Defaults to 1e-8.
            beta_1 (float, optional): Beta1 parameter of Adam. Defaults to 0.
            beta_2 (float, optional): Beta2 parameter of Adam. Defaults to 0.9.
            from_logits (bool, optional): Output range of discriminator, logits imply output on the entire reals. Defaults to True.
        """
        self._paramaters = [training_input, lr_d, lr_g, epsilon, beta_1, beta_2, from_logits]
        self.discriminator = discriminator
        self.generator = generator
        self.noise_shape = [self.generator.input_shape[1], training_input, self.generator.input_shape[-1]]

        self.loss = BinaryCrossentropy(from_logits=from_logits)

        self.generator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def train(self, data, batch_size, n_batches, additional_d_steps):
        """training function of a GAN instance.
        Args:
            data (4d array): Training data in the following shape: (samples, n_series, timesteps, 1).
            batch_size (int): Batch size used during training.
            n_batches (int): Number of update steps taken.
            additional_d_steps (int): Number of extra discriminator training steps during each update.
        """ 
        progress = Progbar(n_batches)

        for n_batch in range(n_batches):
            # sample uniformly
            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]))
            batch = data[batch_idx]

            self.train_step(batch, batch_size, additional_d_steps)

            self.train_hook(n_batch)

            progress.update(n_batch + 1)

    @tf.function
    def train_step(self, data, batch_size, additional_d_steps):

        for _ in range(additional_d_steps + 1):
            noise = tf.random.normal([batch_size, *self.noise_shape])
            generated_data = self.generator(noise, training=False)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(data, training=True)
                fake_output = self.discriminator(generated_data, training=True)
                disc_loss = GAN.discriminator_loss(real_output, fake_output)
            
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, *self.noise_shape])
        generated_data = self.generator(noise, training=False)
        
        noise = tf.random.normal([batch_size, *self.noise_shape])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=False)
            gen_loss = GAN.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train_hook(self, n_batch):
        """Override this method to insert behaviour at every training update.
           Acces to the instance namespace is provided.
        """        
        pass

class CGAN(GAN):
    """ conditonal generative adverserial network class for conditional tcns.
    Training code for a standard DCGAN using the Adam optimizer.
    """    

    def __init__(self, discriminator, generator, training_input, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, from_logits=True):
        """Create a GAN instance
        Args:
            discriminator (tensorflow.keras.models.Model): Discriminator model.
            generator (tensorflow.keras.models.Model): Generator model.
            training_input (int): input size of temporal axis of noise samples.
            lr_d (float, optional): Learning rate of discriminator. Defaults to 1e-4.
            lr_g (float, optional): Learning rate of generator. Defaults to 3e-4.
            epsilon (float, optional): Epsilon paramater of Adam. Defaults to 1e-8.
            beta_1 (float, optional): Beta1 parameter of Adam. Defaults to 0.
            beta_2 (float, optional): Beta2 parameter of Adam. Defaults to 0.9.
            from_logits (bool, optional): Output range of discriminator, logits imply output on the entire reals. Defaults to True.
        """
        super().__init__(discriminator, generator, training_input, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, from_logits=True)

        # build a generator model that left-appends conditional information
        input_layer = Input(shape=self.noise_shape)
        concat_to_output = Input(shape=[self.noise_shape[0]//2, floor(self.noise_shape[1]/2), 1])
        output_generator = self.generator(input_layer)
        output_layer = Concatenate(axis=2)([output_generator, concat_to_output])
        self.cond_generator = Model([input_layer, concat_to_output], output_layer)

    @tf.function
    def train_step(self, data, batch_size, additional_d_steps):

        for _ in range(additional_d_steps + 1):
            noise = tf.concat([tf.zeros([batch_size, self.noise_shape[0]//2, floor(self.noise_shape[1]/2), 3]),
                tf.random.normal([batch_size, self.noise_shape[0]//2, ceil(self.noise_shape[1]/2), 3])], axis=2)

            real_part = tf.repeat(tf.concat([convert_to_tensor(data[:, :, :floor(self.noise_shape[1]/2), :]), 
                tf.zeros([batch_size, self.noise_shape[0]//2, ceil(self.noise_shape[1]/2), 1])], axis=2), 3, axis=3)

            noise_input = tf.concat([real_part, noise], axis=1)

            generated_data = self.cond_generator([noise_input, data[:, :, :floor(self.noise_shape[1]/2), :]],
                training=False)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(data, training=True)
                fake_output = self.discriminator(generated_data, training=True)
                disc_loss = self.discriminator_loss(real_output, fake_output)
            
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.concat([tf.zeros([batch_size, self.noise_shape[0]//2, floor(self.noise_shape[1]/2), 3]),
            tf.random.normal([batch_size, self.noise_shape[0]//2, ceil(self.noise_shape[1]/2), 3])], axis=2)

        real_part = tf.repeat(tf.concat([convert_to_tensor(data[:, :, :floor(self.noise_shape[1]/2), :]), 
            tf.zeros([batch_size, self.noise_shape[0]//2, ceil(self.noise_shape[1]/2), 1])], axis=2), 3, axis=3)

        noise_input = tf.concat([real_part, noise], axis=1)

        with tf.GradientTape() as gen_tape:
            generated_data = self.cond_generator([noise_input, data[:, :, :floor(self.noise_shape[1]/2), :]],
            training=True)
            fake_output = self.discriminator(generated_data, training=False)
            gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.cond_generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.cond_generator.trainable_variables))