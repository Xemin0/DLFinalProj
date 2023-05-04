import tensorflow as tf
import os
from os import listdir
from os.path import join

from PIL import Image



def TrainDatasetFromFolder():
    dataset_dir = "data/DIV2K_train_HR/DIV2K_train_HR/"
    image_filenames = [join(dataset_dir, i) for i in listdir(dataset_dir)]
    lr_images = []
    hr_images = []

    crop_size = 256
    upscale_factor = 4

    crop_size = crop_size - (crop_size % upscale_factor)
    
    for image_filename in image_filenames:
        print(image_filename)
        hr_image = tf.keras.utils.load_img(image_filename)
        hr_image = tf.convert_to_tensor(hr_image)
        print(hr_image)
        
        hr_transform = tf.image.random_crop(hr_image, size = [crop_size,crop_size,3])
        lr_transform = tf.image.resize(hr_transform, crop_size // upscale_factor)
        
        lr_images.append(hr_transform)
        hr_images.append(lr_transform)
        print("hello")
        print(hr_transform)
        return 

    lr_images = tf.stack(lr_images)
    hr_images = tf.stack(hr_images)
    print(lr_images.shape)
    print(lr_images[0].shape)

    return lr_images, hr_images

TrainDatasetFromFolder()





