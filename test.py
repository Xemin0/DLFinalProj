import tensorflow as tf
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



train_num = 500
EPOCHS = 20
BATCH_SIZE = 100

'''
Prepare the Samples for Callback Visualization - Normalized MNIST
'''
# Dataset
# LR (input for Generator); HR = True; SR = (output of Generator)
LR_imgs , HR_imgs = TrainDatasetFromFolder('./Datasets')
SR_imgs = wgan_model0.gen_model(LR_imgs)

# pick 4 pictures for the call back
true_sample = HR_imgs[train_num - 2 : train_num + 2]
fake_sample = SR_imgs[train_num - 2 : train_num + 2]
viz_callback0 = EpochVisualizer(wgan_model0, [true_sample, fake_sample])

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

'''
Visualizing the Results
'''
viz_callback0.save_gif('generated_samples/generationMNIST')
IPython.display.Image(open('generated_samples/generationMNIST.gif', 'rb').read())
