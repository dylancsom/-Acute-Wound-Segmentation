import os
import cv2
import numpy as np
from skimage import exposure

def augment_image_and_mask(image, mask, saturation_factor):
    # Perform horizontal flip
    flipped_image = cv2.flip(image, 1)
    flipped_mask = cv2.flip(mask, 1)
    
    # Adjust saturation of the flipped image
    hsv_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_factor
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return adjusted_image, flipped_mask

# Set the paths to your image and mask folders
image_folder = ""
mask_folder = ""

# Set the augmentation parameters
saturation_factor = 2.0  # Adjust the saturation factor as needed

# Iterate over all images and masks in the folders
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
        image_path = os.path.join(image_folder, filename)
        mask_filename = os.path.splitext(filename)[0] + ".png"  # Update mask filename to have ".png" extension
        mask_path = os.path.join(mask_folder, mask_filename)
        
        # Read the image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image and mask are loaded correctly
        if image is None or mask is None:
            print(f"Skipping {filename} due to loading error.")
            continue
        
        # Perform augmentation on both image and mask
        augmented_image, augmented_mask = augment_image_and_mask(image, mask, saturation_factor)
        
        # Save the augmented image and mask
        augmented_filename = f"augmented_{filename}"
        augmented_image_path = os.path.join(image_folder, augmented_filename)
        augmented_mask_filename = f"augmented_{mask_filename}"
        augmented_mask_path = os.path.join(mask_folder, augmented_mask_filename)
        cv2.imwrite(augmented_image_path, augmented_image)
        cv2.imwrite(augmented_mask_path, augmented_mask)

print("Augmentation completed.")
