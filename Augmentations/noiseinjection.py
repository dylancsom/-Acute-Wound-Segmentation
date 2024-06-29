import os
import cv2
import numpy as np

def augment_image_and_mask(image, mask):
    # Perform 90 degree rotation
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    # Inject noise
    noisy_image = rotated_image.copy()
    noisy_mask = rotated_mask.copy()
    noise = np.random.normal(0, 20, image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)

    return noisy_image, noisy_mask

# Set the paths to your image and mask folders
image_folder = "/content/drive/MyDrive/model.v3.10/images/train"
mask_folder = "/content/drive/MyDrive/model.v3.10/masks/train"

# Iterate over all images and masks in the folders
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
        image_path = os.path.join(image_folder, filename)
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(mask_folder, mask_filename)

        # Read the image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image and mask are loaded correctly
        if image is None or mask is None:
            print(f"Skipping {filename} due to loading error.")
            continue

        # Perform augmentation on both image and mask
        noisy_rotated_image, noisy_rotated_mask = augment_image_and_mask(image, mask)

        # Save the augmented copies
        augmented_filename = f"injection_{filename}"
        augmented_image_path = os.path.join(image_folder, augmented_filename)
        augmented_mask_filename = f"injection_{mask_filename}"
        augmented_mask_path = os.path.join(mask_folder, augmented_mask_filename)
        cv2.imwrite(augmented_image_path, noisy_rotated_image)
        cv2.imwrite(augmented_mask_path, noisy_rotated_mask)

print("Augmentation completed.")