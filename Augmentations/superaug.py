import cv2
import numpy as np
from PIL import Image
import os
import random

# Set the paths
image_dir = "/content/drive/MyDrive/model.v3.10/AugmentTesting/images"
bloodpool_dir = "/content/drive/MyDrive/model.v3.10/AugmentTesting/bloodpool"
mask_dir = "/content/drive/MyDrive/model.v3.10/AugmentTesting/masks"

# Get the list of image files from the "images" directory
image_files = os.listdir(image_dir)

# Get the list of background images from the "bloodpool" directory
background_files = os.listdir(bloodpool_dir)

# Shuffle the background images randomly
random.shuffle(background_files)

# Iterate over each image file
for i, image_file in enumerate(image_files):
    # Load the input image and corresponding mask
    input_image_path = os.path.join(image_dir, image_file)
    input_image = cv2.imread(input_image_path)
    
    mask_file = image_file.split(".")[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, 0)  # Load mask as grayscale
    
    if mask is None:
        print(f"Mask not found for image: {image_file}. Skipping superimposition.")
        continue
    
    # Create a binary mask from the grayscale mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Processing image: {image_file}")
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Binary mask dtype: {binary_mask.dtype}")
    print(f"Binary mask unique values: {np.unique(binary_mask)}")
    
    # Extract the wound segment area from the input image
    wound_segment = cv2.bitwise_and(input_image, input_image, mask=binary_mask)
    
    # Get the corresponding background image from the shuffled list
    background_file = background_files[i % len(background_files)]
    background_image_path = os.path.join(bloodpool_dir, background_file)
    background_image = cv2.imread(background_image_path)
    
    # Create a copy of the background image
    output_image = background_image.copy()
    
    # Superimpose the wound segment onto the background image
    output_image[binary_mask > 0] = wound_segment[binary_mask > 0]
    
    # Save the superimposed image back to the "images" directory with the original file name
    output_file_path = os.path.join(image_dir, image_file)
    cv2.imwrite(output_file_path, output_image)
    print(f"Superimposed image saved to: {output_file_path}")
    print("---")