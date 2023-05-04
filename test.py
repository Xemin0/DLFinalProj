import tensorflow as tf
from tensforflow import keras

import IPython.display

from model.Generator import Generator
#from model.Discriminator import Discriminator
from model.metrics import d_wloss, g_wloss

from data.preprocess import TrainDatasetFromFolder()

from utils.CallBack import EpochVisualizer



'''
Sample Code Snippets to Train and Run a GAN model
'''


'''
Initialize the Network 
'''

wgan_model0 = WGAN(
    dis_model = get_crt_model(),
    gen_model = get_gen_model(),
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



train_num = 50000
EPOCHS = 20

'''
Prepare the Samples for Callback Visualization - Normalized MNIST
'''
# MNIST
true_sample = X0[train_num-2 : train_num+2] ## 4 real images
fake_sample = wgan_model0.z_sampler((4, *wgan_model0.z_dims[1:]))

viz_callback0 = EpochVisualizer(wgan_model0, [true_sample, fake_sample])

# Train the Model
wgan_model0.fit(
    X0[:train_num], L0[:train_num],
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
viz_callback0.save_gif('generationMNIST')
IPython.display.Image(open('generationMNIST.gif', 'rb').read())
