import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class Discriminator(keras.Model):
    '''
    Discriminator for the SRWGAN
    *** No Activation at the Output ***

    Primarily uses
        - LeakyReLU as the activations;
        - Conv2D Blocks
    Input:
        - High/Super-Resolution Image: default shape (256, 256, 3)
                                Range (-1, 1)
    Output:
        - Scalar                (16,16,1) if w\o Faltten()
                                Range (-inf, +inf)
    '''
    def __init__(self, name = 'dis_model', **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)

        self.block1 = Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same"),
                layers.LeakyReLU(),
            ]
        )

        self.block2 = self.__convBlock(64, strides = 2)
        self.block3 = self.__convBlock(128)
        self.block4 = self.__convBlock(128, strides = 2)
        self.block5 = self.__convBlock(256)
        self.block6 = self.__convBlock(256, strides = 2)
        self.block7 = self.__convBlock(512)
        self.block8 = self.__convBlock(512, strides = 2)


        self.linearBlock = Sequential([
            # Flatten for Linear layers
            layers.Flatten(),
            layers.Dense(1024, activation = 'leaky_relu'),
            layers.Dense(512, activation = 'leaky_relu'),
            layers.Dense(1),
        ])

    @staticmethod
    def __convBlock(filters, k_size = 3, strides = 1, alpha = 0.3, padding = 'same'):
        '''Private Static Method - Convolutional Block for reusing internally'''
        return Sequential(
            [
                layers.Conv2D(filters, kernel_size = k_size, strides = strides, padding=padding),
                layers.BatchNormalization(momentum = 0.8),
                layers.LeakyReLU(alpha = alpha),
            ]
        )

    def call(self, inputs):
        '''Forward Call of the Discriminator'''
        block1 = self.block1(inputs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)

        linearBlock = self.linearBlock(block8)
        # No activation needed for Wasserstein Distances
        return linearBlock
