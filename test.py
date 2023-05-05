import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

import IPython.display

from GANcore.WGAN import WGAN
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.metrics import d_wloss, g_wloss

from data.preprocess import TrainDatasetFromFolder

from utils.CallBack import EpochVisualizer


'''
Sample Code Snippets to Train and Run a GAN model
'''


'''
Initialize the Network 
'''
z_dim = 256

wgan_model0 = WGAN(
    dis_model = Discriminator(),
    gen_model = Generator(),
    z_dims = (None, z_dim),
    name = 'wgan'
)

print('1')

wgan_model0.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
        'g_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
    },
    losses = {
        'd_loss' : d_wloss,
        'g_loss' : g_wloss,
    },
    accuracies = {}
)

print('2')

train_num = 20
EPOCHS = 2
BATCH_SIZE = 4

'''
Prepare the Samples for Callback Visualization - Normalized MNIST
'''
# Dataset
# LR (input for Generator); HR = True; SR = (output of Generator)
LR_imgs , HR_imgs = TrainDatasetFromFolder('./Datasets')
print('3')
LR_imgs = (LR_imgs-127.5)/127.5
HR_imgs = (HR_imgs-127.5)/127.5

SR_imgs = wgan_model0.gen_model(LR_imgs)
print('4')
# pick 4 pictures for the call back
true_sample = HR_imgs[train_num - 2 : train_num + 2]
fake_sample = SR_imgs[train_num - 2 : train_num + 2]
print('5')
viz_callback0 = EpochVisualizer(wgan_model0, [true_sample, fake_sample])

print('6')

# Train the Model
wgan_model0.fit(
    LR_imgs[:train_num], HR_imgs[:train_num],
    dis_steps = 3,
    gen_steps = 1,
    gp_weight = 10.0,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [viz_callback0]
)

## Either Save the model/Visualizer
## or directly visualize the CallBack
print('7')
'''
Visualizing the Results
'''
viz_callback0.save_gif('generated_samples/generationMNIST')
IPython.display.Image(open('generated_samples/generationMNIST.gif', 'rb').read())
