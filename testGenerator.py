'''
Sample Code Snippets for:
Testing the pretrained Generator on
    1. Single image
    2. GIF image

Requires APIs from ./utils/testGenerate.py
'''

from utils.testGenerate import load_gen, load_img, generate_one_img, generate_gif

import matplotlib.pyplot as plt

## Load Pretrained Model

## Generator Requires Batched Images Inputs of:
##    - Shape (B, 64, 64, C) # Ideally RGB channels, thus C = 3
##    - Range (-1, 1)

test_Gen = load_gen() # Default path = './SRWGAN_Model'


'''
Test on Generating an Image
'''
# Load A Single Image
img = load_img(
               imname = 'ji1.png',
               base_path = './generate_samples',  # Default
               resize = (64, 64),        # Cropped to square while preserving the center
                                         # Resized to (64, 64)
               centralize = False        # Default, 
                                         # False: pixel values range = (0.0, 1.0); 
                                         # True : pixel values range = (-1.0,1.0)
              )

# test the model on the loaded image
# Required input image shape: (64, 64, C)
#                      range: [0.0, 1.0] 
sr_img = generate_one_img(test_Gen, img)

# Show the Loaded (Cropped and Resized) Image along with the Super-Resoluted Image
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 10))
fig.suptitle(f'Down Sampled Image of Shape {img.shape} v.s. Super-Resed Image of Shape {sr_img.shape}')
axes[0].imshow(img)
axes[0].set_title('Down Sampled Image')

axes[1].imshow(sr_img)
axes[1].set_title('Supter-Resed Image')


# Save the images



'''
Test on Generating an GIF
'''
## Both the (Cropped and) Resized GIF and the Super-Res GIF will be saved into the same path where the original GIF is loaded

generate_gif(
    gen_model = test_Gen,
    src_filename = 'test.gif',
    base_path = './generated_samples',    # Default
    resize = (64, 64),                    # Default
    batch_size,                           # Default; Pass in images in batches to avoid OOM and large overhead
)
