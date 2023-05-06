import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class Generator(keras.Model):
    '''
    Generator for the GAN

    Primarily uses
        - Parametric ReLU as the activations;
        - Sub-Pixel Convolution for UpSampling
    Input:
        - Low-Resolution Image: default shape (64, 64, 3)
                                Range (-1, 1)
        - High-Resolution Image: default shape (256, 256, 3)
                                Range (-1, 1)
    '''
    def __init__(self, scale_factor = 4, name = 'gen_model', **kwargs):
        super(Generator, self).__init__(name = 'gen_model', **kwargs)

        # UpSample Block Number for Sub-Pixel Convolutions
        #scale_factor = 4
        upsample_block_num = int(tf.math.log(scale_factor + 0.0) / tf.math.log(2.0))

        self.block1 = Sequential(
            [
                layers.Conv2D(64, kernel_size=9, padding="same"),
                layers.PReLU(),
            ]
        )

        self.block2 = self.__convBlock()
        self.block3 = self.__convBlock()
        self.block4 = self.__convBlock()
        self.block5 = self.__convBlock()
        self.block6 = self.__convBlock()
        self.block7 = self.__convBlock()
        self.block8 = self.__convBlock()
        self.block9 = self.__convBlock()
        self.block10 = self.__convBlock()
        self.block11 = self.__convBlock()
        self.block12 = self.__convBlock()
        self.block13 = self.__convBlock()
        self.block14 = self.__convBlock()
        self.block15 = self.__convBlock()
        self.block16 = self.__convBlock()
        self.block17 = self.__convBlock()

        self.block18 = Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
            ]
        )

        # Sub-Pixel UpSampling
        self.block19 = Sequential(
            [
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=upsample_block_num)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=upsample_block_num)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=upsample_block_num)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=upsample_block_num)),
                layers.PReLU(),
                layers.Conv2D(3, kernel_size=9, padding="same", activation='tanh')
                ## At the output layer `tanh` restric the results between [-1, 1]
            ]
        )

    @staticmethod
    def __convBlock():
        '''Private Static Method - Convolutional Block for reusing internally'''
        return Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Conv2D(64, kernel_size=9, padding="same"),
                layers.PReLU(),
                layers.BatchNormalization(),
            ]
        )

    def call(self, inputs):
        '''Forward Call of the Model'''
        block1 = self.block1(inputs)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block4(block6)
        block8 = self.block5(block7)
        block9 = self.block6(block8)
        block10 = self.block4(block9)
        block11 = self.block5(block10)
        block12 = self.block6(block11)
        block13 = self.block4(block12)
        block14 = self.block5(block13)
        block15 = self.block6(block14)
        block16 = self.block4(block15)
        block17 = self.block5(block16)

        block18 = self.block18(block17)
        block19 = self.block19(block1 + block18)

        ### ** Rescaling not necessary inside the Generator **
        #return (tf.math.tanh(block19) + 1) / 2
        return block19


