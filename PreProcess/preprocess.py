import os
import numpy as np
import tensorflow as tf

import tensorflow as tf
from keras.backend import clear_session


# Define a function to apply augmentation to images and masks
tf.keras.backend.clear_session()

def get_mask_path(project_dir, set_name, image_file_name):
    dataset_dir = os.path.join(project_dir, set_name)
    mask_folder_dir = os.path.join(dataset_dir, 'masks')  # The folder specifically for masks
    mask_filename = f"{os.path(image_file_name)}.png"  # The mask file in PNG format
    mask_path = os.path.join(mask_folder_dir, mask_filename)
    return mask_path
   




def image_mask_generator(image_dir, mask_dir, batch_size, target_size):
    from keras.preprocessing.image import load_img, img_to_array
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    mask_filenames = {os.path.basename(f): os.path.join(mask_dir, os.path.basename(f)) for f in image_filenames}
    image_paths = [os.path.join(image_dir, f) for f in image_filenames]
    
    total_files = len(image_filenames)
    loaded_files = 0  # Track how many files have been loaded
    while loaded_files < total_files:
        batch_images = []
        batch_masks = []
        for i in range(loaded_files, min(loaded_files + batch_size, total_files)):
            img_path = image_paths[i]
            mask_path = mask_filenames[os.path.basename(img_path)]
            
            img = load_img(img_path, target_size=target_size, color_mode='rgb')
            img = img_to_array(img) / 255.0  # Rescale to [0, 1]
            
            mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')
            mask = img_to_array(mask) / 255.0  # Rescale to [0, 1]
            
            batch_images.append(img)
            batch_masks.append(mask)
        
        loaded_files += len(batch_images)
        yield np.array(batch_images), np.array(batch_masks)





# Example usage
project_directory = '/Users/dylansomra/Desktop/model.v3.10'
batch_size = 3
target_size = (256, 256)

train_images_path = '/Users/dylansomra/Desktop/model.v3.10/images/train'
train_masks_path = '/Users/dylansomra/Desktop/model.v3.10/masks/train'
val_images_path = '/Users/dylansomra/Desktop/model.v3.10/images/val'
val_masks_path = '/Users/dylansomra/Desktop/model.v3.10/masks/val'

# Create a generator using the modified function
train_dataset = tf.data.Dataset.from_generator(
    lambda: image_mask_generator(train_images_path, train_masks_path, batch_size, target_size),
    output_signature=(
        tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *target_size, 1), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: image_mask_generator(val_images_path, val_masks_path, batch_size, target_size),
    output_signature=(
        tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *target_size, 1), dtype=tf.float32)
    )
)



# Check data loading and consistency

