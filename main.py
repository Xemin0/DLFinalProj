import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        scale_factor = 4
        upsample_block_num = int(tf.math.log(scale_factor) / tf.math.log(2))
        
        self.block1 = Sequential(
            [
                layers.Conv2D(64, kernel_size=9, padding="same"),
                layers.PReLU(),
            ]
        )

        RB = Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Conv2D(64, kernel_size=9, padding="same"),
                layers.PReLU(),
                layers.BatchNormalization(),
            ]
        )

        self.block2 = RB
        self.block3 = RB
        self.block4 = RB
        self.block5 = RB
        self.block6 = RB
        self.block7 = RB
        self.block8 = RB
        self.block9 = RB
        self.block10 = RB 
        self.block11 = RB
        self.block12 = RB
        self.block13 = RB
        self.block14 = RB
        self.block15 = RB
        self.block16 = RB
        self.block17 = RB

        self.block18 = Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
            ]
        )

        self.block19 = Sequential(
            [
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2)),
                layers.PReLU(),
                layers.Conv2D(64 * 2 ** 2, kernel_size=3, padding="same"),
                layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2)),
                layers.PReLU(),
                layers.Conv2D(3, kernel_size=9, padding="same")
            ]
        )


    def call(self, inputs):
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

        return (tf.math.tanh(block19) + 1) / 2



