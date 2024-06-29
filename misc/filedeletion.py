import os
import shutil

# Set the paths to your image and mask folders
image_folder = "/content/drive/MyDrive/model.v3.10/AugmentTesting/images"
mask_folder = "/content/drive/MyDrive/model.v3.10/AugmentTesting/masks"

# Function to delete files with a given prefix
def delete_files_with_prefix(folder, prefix):
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {file_path}, {e}")

# Delete augmented files from the image folder
print("Deleting augmented files from the image folder...")
delete_files_with_prefix(image_folder, "augmented_")

# Delete augmented files from the mask folder
print("\nDeleting augmented files from the mask folder...")
delete_files_with_prefix(mask_folder, "augmented_")

print("\nDeletion completed.")