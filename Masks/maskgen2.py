import os
import cv2
import json
import numpy as np
import shutil

source_folder = "/content/drive/MyDrive/newdata"
json_path = "/content/drive/MyDrive/newdata/via_project_28Mar2024_16h0m_json-2.json"

count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask

# Read JSON file
with open(json_path) as f:
    data = json.load(f)

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key):
    try:
        regions = data[itr]["regions"]
        for region in regions:
            x_points = region["shape_attributes"]["all_points_x"]
            y_points = region["shape_attributes"]["all_points_y"]
            all_points = []
            for i, x in enumerate(x_points):
                all_points.append([x, y_points[i]])
            file_bbs[key] = all_points
    except (KeyError, IndexError, TypeError):
        print("No bounding box. Skipping", key)

for itr in data:
    file_name_json = data[itr]["filename"]
    add_to_dict(data, itr, file_name_json[:-4])

print("\nDict size: ", len(file_bbs))

for file_name in file_bbs:
    to_save_folder = os.path.join(source_folder, file_name)
    image_folder = os.path.join(to_save_folder, "images")
    mask_folder = os.path.join(to_save_folder, "masks")

    # Create folders if they don't exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Copy image to new location
    curr_img = os.path.join(source_folder, file_name + ".jpg")
    if not os.path.exists(curr_img):
        curr_img = os.path.join(source_folder, file_name + ".jpeg")
    shutil.copy(curr_img, os.path.join(image_folder, os.path.basename(curr_img)))

    # Generate mask and save in corresponding folder
    if file_name in file_bbs:
        # Read the original image to get its dimensions
        image_path = os.path.join(image_folder, os.path.basename(curr_img))
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Create mask with the same dimensions as the original image
        mask = np.zeros((height, width), dtype=np.uint8)
        arr = np.array(file_bbs[file_name], dtype=np.int32)
        count += 1
        cv2.fillPoly(mask, [arr], color=255)
        cv2.imwrite(os.path.join(mask_folder, file_name + ".png"), mask)

print("Images saved:", count)