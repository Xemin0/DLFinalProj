import tensorflow as tf
from tensorflow import keras

import IPython.display


from GANcore.SRWGAN import SRWGAN
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.metrics import d_srloss, g_srloss

from data.preprocess import TrainDatasetFromFolder

from utils.CallBack import EpochVisualizer



'''
Sample Code Snippets to Train and Run a GAN model
'''


'''
Initialize the Network
'''


srwgan_model = SRWGAN(
    dis_model = Discriminator(),
    gen_model = Generator(),
    name = 'srwgan',
    z_dims = [None, None],  # Not using random samples for Generator's Input in SRWGAN 
    # Default Values
    pretrained = 'resnet50',
    hyperimg_ids = [2, 7, 10, 14],
    lr_shape = [64, 64, 3],     # Low-Res Images as the input for Generator
    hr_shape = [256, 256, 3]    # High-Res Images as the ground Truth; 
)                               # Super-Res Images as the output for Generator 


srwgan_model.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
        'g_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
    },
    losses = {
        'd_loss' : d_srloss,
        'g_loss' : g_srloss,
    },
)


'''
Load the Data for Super Resolution Task
'''
lres, hres = TrainDatasetFromFolder(dataset_dir = './Datasets')

# Centralize the data to [-1, 1] to for better training (and ofc to avoid overhead and overflow in memory)
lres = (lres - 127.5) / 127.5
hres = (hres - 127.5) / 127.5

train_num = 20
EPOCHS = 2
BATCH_SIZE = 4

'''
Prepare the Samples for Callback Visualization - Centralized + Normalized
'''
true_sample = hres[train_num-2 : train_num+2]                         ## 4 High Resolution images
fake_sample = srwgan_model.gen_model(lres[train_num-2 : train_num+2])   ## 4 Generated Super Resolution images
print('high-res samples shape:', true_sample.shape)
print('generatered super-res samples shape:', fake_sample.shape)

viz_callback = EpochVisualizer(srwgan_model, [true_sample, fake_sample])

# Train the Model
srwgan_model.fit(
    lres[:train_num], hres[:train_num],
    dis_steps = 3,
    gen_steps = 1,
    gp_weight = 10.0,
    content_weight = 1e-3,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [viz_callback]
)

## Either Save the model/Visualizer
## or directly visualize the CallBack

'''
Visualizing the Results
'''
viz_callback0.save_gif('generatedSuperRes')
IPython.display.Image(open('generatedSuperRes.gif', 'rb').read())

