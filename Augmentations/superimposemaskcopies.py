import os
import shutil

# Set the directory paths
images_dir = "/content/drive/MyDrive/model.v3.10/AugmentTesting/images"
masks_src_dir = "/content/drive/MyDrive/model.v3.10/masks/train"
masks_dest_dir = "/content/drive/MyDrive/model.v3.10/AugmentTesting/masks"

# Create the destination directory if it doesn't exist
os.makedirs(masks_dest_dir, exist_ok=True)

# Loop through all files in the images directory
for filename in os.listdir(images_dir):
    # Check if the file is a JPEG image
    if filename.endswith(".jpg"):
        # Split the filename into base and extension
        base, _ = os.path.splitext(filename)
        
        # Remove the "superimpose_" prefix from the base
        original_base = base[12:]
        
        # Construct the source and destination mask paths
        mask_src_path = os.path.join(masks_src_dir, original_base + ".png")
        mask_dest_path = os.path.join(masks_dest_dir, filename[:-4] + ".png")
        
        # Check if the source mask file exists
        if os.path.isfile(mask_src_path):
            # Copy the mask file to the destination directory
            shutil.copy(mask_src_path, mask_dest_path)
            print(f"Copied '{original_base}.png' to '{mask_dest_path}'")
        else:
            print(f"Mask file not found for '{original_base}'")