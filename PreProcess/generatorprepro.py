import numpy as np
import tensorflow as tf
from functools import partial
import os
import PIL
def image_mask_generator(image_dir, mask_dir, batch_size, target_size=(256, 256), input_format='jpg', mask_format='png'):
    print("Generator function is being executed.")
    from keras.preprocessing.image import img_to_array, load_img
    
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(f'.{input_format}')]
    mask_filenames = {os.path.basename(f): os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + f'.{mask_format}') for f in image_filenames}
    image_paths = [os.path.join(image_dir, f) for f in image_filenames]
    
    total_files = len(image_filenames)
    loaded_files = 0  # Track how many files have been loaded
    print("Image Files:", image_filenames)
    print("Mask Files:", mask_filenames)
    while loaded_files < total_files:
        batch_images = []
        batch_masks = []
        for i in range(loaded_files, min(loaded_files + batch_size, total_files)):
            img_path = image_paths[i]
            mask_path = mask_filenames[os.path.basename(img_path)]
            
            img = load_img(img_path, target_size=target_size, color_mode='rgb')
            img = img_to_array(img) / 255.0  # Rescale to [0, 1]
            
            mask = load_img(mask_path, target_size=target_size, color_mode='grayscale' if mask_format == 'png' else 'rgb')
            mask = img_to_array(mask) / 255.0  # Rescale to [0, 1]
            
            batch_images.append(img)
            batch_masks.append(mask)
        
        loaded_files += len(batch_images)
        yield np.array(batch_images), np.array(batch_masks)



batch_size = 2
target_size = (256, 256)
train_images_path = '/content/drive/MyDrive/model.v3.10/images/train'
train_masks_path = '/content/drive/MyDrive/model.v3.10//masks/train'
val_images_path = '/content/drive/MyDrive/model.v3.10/images/val'
val_masks_path = '/content/drive/MyDrive/model.v3.10/masks/val'

train_generator = partial(image_mask_generator, batch_size=batch_size, target_size=(256, 256), input_format='jpg', mask_format='png')
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator(train_images_path, train_masks_path),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
    )
)
val_generator = partial(image_mask_generator, batch_size=batch_size, target_size=(256, 256), input_format='jpg', mask_format='png')
val_dataset = tf.data.Dataset.from_generator(
    image_mask_generator,
    args=(val_images_path, val_masks_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
    )
)



total_files = len(os.listdir(train_images_path))


# Print some information about the data
