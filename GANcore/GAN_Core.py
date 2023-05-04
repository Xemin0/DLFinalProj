import tensorflow as tf
from tensorflow.keras import layers

class GAN_Core(tf.keras.Model):

    '''
    Base class to administrate both the Generator and Discriminator
    '''
    def __init__(self, dis_model, gen_model, z_dims, z_sampler=tf.random.normal, **kwargs):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super().__init__(**kwargs)
        self.z_dims = z_dims
        self.z_sampler = z_sampler
        self.gen_model = gen_model
        self.dis_model = dis_model

    '''
    New calling API for the model
    '''
    def sample_z(self, num_samples, **kwargs):
        '''generates an z based on the z sampler'''
        return self.z_sampler([num_samples, *self.z_dims[1:]])

    def discriminate(self, inputs, **kwargs):
        '''predict whether input input is a real entry from the true dataset'''
        return self.dis_model(inputs, **kwargs)

    def generate(self, z, **kwargs):
        '''generates an output based on a specific z realization'''
        return self.gen_model(z, **kwargs)

    '''
    ** keras.Model will handle all components defined in its end-to-end architecture **
    Link the Generator and Discriminator within the GAN:
    '''
    def call(self, inputs, **kwargs):
        b_size = tf.shape(inputs)[0]

        z_samp = self.sample_z(b_size)   ## Generate a z sample
        g_samp = self.generate(z_samp)   ## Generate an x-like image
        d_samp = self.discriminate(g_samp)   ## Predict whether x-like is real
        print(f'Z( ) Shape = {z_samp.shape}')
        print(f'G(z) Shape = {g_samp.shape}')
        print(f'D(x) Shape = {d_samp.shape}\n')
        return d_samp

    def build(self, input_shape, **kwargs):
        super().build(input_shape=self.z_dims, **kwargs)
