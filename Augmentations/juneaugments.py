import os
import cv2
import numpy as np
from skimage import exposure

def augment_image_and_mask(image, mask):
    # Perform horizontal flip
    flipped_image = cv2.flip(image, 1)
    flipped_mask = cv2.flip(mask, 1)
    
    # Perform 90 degree rotation
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    
    # Inject noise
    noisy_image = image.copy()
    noise = np.random.normal(0, 20, image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    
    
    
    return flipped_image, flipped_mask, rotated_image, rotated_mask, noisy_image, mask

# Set the paths to your image and mask folders
image_folder = "/content/drive/MyDrive/model.v3.10/newimages"
mask_folder = "/content/drive/MyDrive/model.v3.10/newmasks"

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
        
        # Perform augmentations on both image and mask
        noisy_image, noisy_mask, flipped_contrast_image, flipped_contrast_mask = augment_image_and_mask(image, mask)
        
  
        
        
        
        noisy_filename = f"noisy_{filename}"
        noisy_image_path = os.path.join(image_folder, noisy_filename)
        noisy_mask_filename = f"noisy_{mask_filename}"
        noisy_mask_path = os.path.join(mask_folder, noisy_mask_filename)
        cv2.imwrite(noisy_image_path, noisy_image)
        cv2.imwrite(noisy_mask_path, noisy_mask)
        
        

print("Augmentation completed.")