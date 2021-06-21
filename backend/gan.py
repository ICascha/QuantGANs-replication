from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model

import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class GAN:
    """ Generative adverserial network class.

    Training code for a standard DCGAN using the Adam optimizer.
    Code taken in part from: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
    """    

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = GAN.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = GAN.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return GAN.cross_entropy(tf.ones_like(fake_output), fake_output)

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

    def save(self, folder_path):
        """save the GAN instance, including training variables

        Args:
            folder_path (string): path of folder to store GAN setup.
        """        
        # see: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        symbolic_weights = getattr(self.generator_optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('optimizer.pkl', 'wb') as f:
            pickle.dump(folder_path + '/w_G', f)

        symbolic_weights = getattr(self.discriminator_optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('optimizer.pkl', 'wb') as f:
            pickle.dump(folder_path + '/w_D', f)

        with open('params', 'wb') as f:
            pickle.dump(folder_path + '/params', f)
        
        self.generator.save(folder_path + '/G')
        self.discriminator.save(folder_path + '/D')

    @staticmethod
    def load(folder_path):
        """load a stored gan instance

        Args:
            folder_path (string): path of folder to store GAN setup.

        Returns:
            GAN: GAN instance containing previous training variables.
        """        
        generator = load_model(folder_path + '/G')
        discriminator = load_model(folder_path + '/D')

        # see: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        model.load_weights('weights.h5')
        with open(folder_path + '/w_G/optimizer.pkl', 'rb') as f:
            w_G = pickle.load(f)
        with open(folder_path + '/w_D/optimizer.pkl', 'rb') as f:
            w_D = pickle.load(f)

        with open(folder_path + '/params', 'rb') as f:
            params = pickle.load(f)

        return GAN(generator, discriminator, *params)

    def train_hook(self, n_batch):
        """Override this method to insert behaviour at every training update.
           Acces to the instance namespace is provided.
        """        
        pass