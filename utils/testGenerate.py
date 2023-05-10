'''
Subroutines for testing Saved Generator Model
** Added GIF support
'''

import tensorflow as tf

import numpy as np

# for GIF
import imageio
from pygifsicle import optimize

from os.path import join
import matplotlib.pyplot as plt


def load_gen(path = './SRWGAN_Gen'):
    '''Load the Saved Generator'''
    return tf.keras.models.load_model(path)


'''
Processing Images
'''
def load_img(imname, base_path = './generated_samples', resize = None, centralize = False):
    '''
    Load images from the specified path
    inputs:
        - imname    : File name of the image to be loaded
        - base_path:
        - resize    : Whether to resize the image;
                      specified  : resize img to the given size; Cropped to smallest side, while keeping the center
                      not specifie (None): return the loaded image
        - centralize: Whether to centralize the image pixel values
                      True : range (-1.0, 1.0) as the Generator's input
                      False: range (0.0, 1.0) for plotting
    return:
        - img      : return the (resized) img of shape (H, W, C)
                                 pixel values of range (min, max)
    '''
    # Load the image and convert to tensor with dytpe = float
    img = tf.keras.utils.load_img(join(base_path, imname))
    img = tf.convert_to_tensor(img, dtype = tf.float32)

    # whether to centralize the pixel values
    img = (img - 127.5) / 127.5    # (-1.0, 1.0)
    if not centralize:
        img = (img + 1.0) / 2.0    # (0.0, 1.0)

    # whether to resize the image to the specified size
    if resize is not None:  #### ** Add padding to avoid loss of ratio (distortion)
        #img = tf.image.resize(img, size = resize, preserve_aspect_ratio = True)
        #img = tf.image.resize_with_pad(img, resize[0], resize[1])

        # Crop the image to a square based on the smallest side
        # While keep the center
        img = crop2square(img)

        # Resize
        img = tf.image.resize(img, size = resize[:2])

    return img

def crop2square(img):
    '''
    crop a given image to square based on the smallest side 
    while keeping the center
    img could be either of shape :
        (H, W, C)  or
        (B, H, W, C)
    '''
    H, W = img.shape[-3:-1]
    if H < W:   # Crop along the W
        Wc = W // 2
        if 3 == img.ndim:
            img = img[:, Wc - (H // 2) : Wc + (H - H // 2), :]
        elif 4 == img.ndim:
            img = img[:, :, Wc - (H // 2) : Wc + (H - H // 2), :]
        else:
            raise Exception('Input numpy array has to be of shape (H, W, C) or (B, H, W, C)')
    elif H > W: # Crop along the H
        Hc = H // 2
        if 3 == img.ndim:
            img = img[Hc - (W // 2) : Hc + (W - W//2), :, :]
        elif 4 == img.ndim:
            img = img[:, Hc - (W // 2) : Hc + (W - W//2), :, :]
        else:
            raise Exception('Input numpy array has to be of shape (H, W, C) or (B, H, W, C)')
    return img

def generate_one_img(gen_model, img):
    '''
    Use a Gen_model to generate Super Resolution img
    input:
        gen_model:  Requires batched inputs of shape (batchsz, H = 64, W = 64, C); 
                    ** So here we will use tf.expand_dims(img, axis = 0)
        img      :  Single image input of size (H = 64, W = 64, C)
                    numpy.array or tf.Tensor of range [0, 1]

    output:
        super-res: tf.Tensor of range (0, 1) to be able to plot directly
    '''
    img = img * 2.0 - 1.0
    img = gen_model(tf.expand_dims(img, axis = 0))
    img = (img + 1.0) / 2.0
    return img[0]


def generate_batches(gen_model, imgs, batch_size = 10):
    '''
    Forward Pass of the Model as Batches to Avoid OOM and Large Overhead
    ** Assuming the imgs are already preprocessed to the proper size and range **
    ** to be fed into the model **
    '''
    assert 4 == len(imgs.shape), f"Use 'generate_one_img' instead if the input is not batched\nReceived dim = {len(imgs.shape)}"
    num = imgs.shape[0]
    n_batches = num // batch_size
    remainder = bool(num % batch_size)

    print(f'Received a total number of {num} images, and {batch_size=}\nThe number of batches to process:\t{n_batches + remainder}')
    # forward pass of the model in batches

    '''
    outputs = [gen_model(imgs[i*batch_size : (i+1)*batch_size]) for i in range(n_batches)]
    # the remainder batch
    outputs += [gen_model(imgs[n_batches * batch_size : ])]
    '''

    outputs = []
    for i in range(n_batches):
        curr_batch = imgs[i*batch_size : (i+1)*batch_size]
        output_batch = gen_model(curr_batch)
        outputs += [output_batch]
        if n_batches - 1 == i and remainder: # The remainder batch
            outputs += [gen_model(imgs[n_batches * batch_size : ])]
        print(f'\rProcessing Batch: [{i+1} / {n_batches + remainder}]', end='')



    print('\n=============== Finished Processing all batches!! ===============')
    # Concatenate along the batch_dimension
    outputs = tf.concat(outputs, axis = 0)
    return outputs



'''
Processing GIF
'''
def load_gif(filename, base_path = './generated_samples'):
    '''
    Inputs:
        - filename :
        - base_path:

    Outputs:
        - frames: frames of images as numpy arrays stacked along the first dimension
                  shape (num_frames, H, W, C)
        - duration: duration of each frame in miliseconds
        - loop: whether the gif is looped
    '''
    frames = []
    duration = None
    loop = None
    # Load the GIF
    if filename.split('.')[-1] not in ('gif', 'GIF'):
        filename = filename + '.gif'
    frames = imageio.v3.imread(join(base_path, filename))
    print('Loading GIF from:')
    print(join(base_path, filename))
    print('==================================================')
    duration, loop = gif_info(filename, base_path)
    return frames, duration, loop



def save_gif(frames, filename, duration = 40, loop = False, base_path = './generated_samples'):
    '''
    Save the given frames to gif at specified location; and optimize it to reduce the size
    Inputs:
        - frames : list of (H, W, C) images or stacked (batchsz, H, W, C) numpy arrays
    '''
    if filename.split('.')[-1] not in ('gif', 'GIF'):
        filename = filename + '.gif'
    if type(frames) == list:
        frames = np.stack(frames, axis = 0)

    if frames.max() <= 1.0:
        frames = (frames * 255)
    frames = frames.astype(np.uint8)
    # Requires frames to be of np.uint8 dtype
    imageio.v3.imwrite(join(base_path, filename), frames, duration = duration, loop = loop)
    # optimize to reduce the size
    optimize(join(base_path, filename))

def gif_info(filename, base_path = './generated_samples'):
    '''
    Get the 'loop' and 'duration' information of a gif for reconstructing the gif
    '''
    with imageio.get_reader(join(base_path, filename)) as reader:
        meta_data = reader.get_meta_data()
        loop = meta_data.get('loop', 0)
        duration = meta_data.get('duration', 0)
    return duration, loop



def generate_gif(gen_model, src_filename, resize = (64, 64), batch_size = 10, base_path = './generated_samples'):
    '''
    1. Load the source gif from specified path
    2. Crop and resize to size resize = (64, 64)
    3. Apply the Super-Resolution Generator to each frames of the GIF
    4. Save and optimize the Cropped and Resized GIF
    4. Save and optimize the Super_Resed GIF

    inputs:
        - gen_model
        - src_filename
        - trg_filename
        - resize    : default as size(64, 64)
        - base_path :
    '''
    # Process the filenames
    if src_filename.split('.')[-1] not in ('gif', 'GIF'):
        trg_filename = src_filename + '_SR.gif'
        src_filename += '.gif'
        resized_src_filename += src_filename + '_resized.gif'
    else:
        trg_filename = '.'.join(src_filename.split('.')[:-1]) + '_SR.gif'
        resized_src_filename = '.'.join(src_filename.split('.')[:-1]) + '_resized.gif'



    # Load the GIF and get necessary information;
    # frames.dtype = np.uint8
    frames, duration, loop = load_gif(src_filename, base_path = base_path)
    # Crop the image to a square and resize to a desired size (64, 64) by default
    frames = crop2square(frames)
    frames = tf.image.resize(frames, size = resize[:2])

    # Save the Resized GIF
    resized_frames = frames._numpy().astype(np.uint8)
    print(f'Saving resized (newsize = {resize}) GIF to :\n{join(base_path,resized_src_filename)}')
    print('==================================================')

    #imageio.v3.imwrite(join(base_path, resized_src_filename), resized_frames, duration = duration, loop = loop)
    #optimize(join(resized_src_filename, base_path))

    print('Resized Images of Shape:\t', resized_frames.shape)

    save_gif(resized_frames, filename = resized_src_filename,\
             duration = duration, loop = loop, base_path = base_path)

    # Centralize the pixel values of the batched images (as a numpy array)
    # to range (-1, 1)
    frames = (frames - 127.5) / 127.5

    ########## feed to the Generator ##########
    #*** Need to feed in as batches to avoid OOM and large overhead ***#
    SR_frames = generate_batches(gen_model, frames, batch_size = batch_size)

    # Decentralize
    SR_frames = ((SR_frames * 127.5) + 127.5)._numpy().astype(np.uint8)

    print('Super-Resed Images of Shape:\t', SR_frames.shape)

    # Save the Super-Resed GIF
    print(f'Saving Super-Resed GIF to :\n{join(base_path,trg_filename)}')
    print('==================================================')

    #imageio.v3.imwrite(join(base_path, trg_filename), SR_frames, duration = duration, loop = loop)
    #optimize(join(base_path, trg_filename))
    save_gif(SR_frames, filename = trg_filename,\
             duration = duration, loop = loop, base_path = base_path)
