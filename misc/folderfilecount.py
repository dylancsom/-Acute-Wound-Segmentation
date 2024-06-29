import os

def count_files(directory):
    total_files = len(os.listdir(directory))
    print(f"Total files in {directory}: {total_files}")

# Specify the directories for images and masks
image_directory = '/content/drive/MyDrive/model.v3.10/images/train'
mask_directory = '/content/drive/MyDrive/model.v3.10/masks/train'

print("Counting files in the images folder...")
count_files(image_directory)

# Count files in the masks folder
print("Counting files in the masks folder...")
count_files(mask_directory)

print("File count complete.")