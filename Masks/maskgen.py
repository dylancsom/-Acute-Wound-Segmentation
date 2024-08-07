import os
import cv2
import json
import numpy as np
source_folder = "/content/drive/MyDrive/newdata2"
json_path = "/content/drive/MyDrive/newdata2/via_project_28Mar2024_16h0m_json.json"


count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 256  # Dimensions should match those of ground truth image
MASK_HEIGHT = 256

# Read JSON file
with open(json_path) as f:
    data = json.load(f)

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except KeyError:
        print("No bounding box. Skipping", key)
        return

    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    file_bbs[key] = all_points

for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 0  # Contains count of masks for a single ground truth image
    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count + 1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nDict size: ", len(file_bbs))

for file_name in os.listdir(source_folder):
    to_save_folder = file_name[:-4]
    image_folder = os.path.join(to_save_folder, "newdata")
    mask_folder = os.path.join(to_save_folder, "newmasks")
    curr_img = os.path.join(source_folder, file_name)

    # Create folders if they don't exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Copy image to new location
    os.rename(curr_img, os.path.join(image_folder, file_name))

    # For each entry in dictionary, generate mask and save in corresponding folder
    for itr in file_bbs:
        num_masks = itr.split("*")
        mask_folder = os.path.join(to_save_folder, "newmasks")
        mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))

        try:
            arr = np.array(file_bbs[itr])
        except:
            print("Not found:", itr)
            continue

        count += 1
        cv2.fillPoly(mask, [arr], color=(255))

        if len(num_masks) > 1:
            cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".png"), mask)
        else:
            cv2.imwrite(os.path.join(mask_folder, itr + ".png"), mask)

print("Images saved:", count)