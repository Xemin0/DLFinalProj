import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class Discriminator(keras.Model):
    '''
    Discriminator for the GAN

    Primarily uses
        - Leaky ReLU as the activations;
        - Convolution and Pooling
    '''
    def __init__(self, name='discriminator', **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        
        self.block1 = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.block2 = Sequential([
            layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.block3 = Sequential([
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.block4 = Sequential([
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        '''Forward Call of the Model'''
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = layers.Flatten()(x)
        x = self.output_layer(x)
        return x