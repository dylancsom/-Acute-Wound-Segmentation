import os

def check_corresponding_masks(image_directory, mask_directory):
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_directory) if f.lower().endswith('.png')]

    missing_masks = []
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        corresponding_mask = f"{image_name}.png"
        if corresponding_mask not in mask_files:
            missing_masks.append(image_file)

    if missing_masks:
        print("The following images are missing corresponding mask files:")
        for missing_mask in missing_masks:
            print(missing_mask)
    else:
        print("All images have corresponding mask files.")

# Specify the directories for images and masks
image_directory = '/path/to/images/folder'
mask_directory = '/path/to/masks/folder'

# Check for corresponding masks
print("Checking for corresponding mask files...")
check_corresponding_masks(image_directory, mask_directory)