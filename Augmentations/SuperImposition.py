import cv2
import numpy as np
from PIL import Image

# Load the input image and mask
input_image_path = ""
input_mask_path = ""
background_image_path = ""
output_dir = ""

input_image = cv2.imread(input_image_path)
input_mask = cv2.imread(input_mask_path, 0)  # Load mask as grayscale
background_image = cv2.imread(background_image_path)

# Create a binary mask from the grayscale mask
_, binary_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)

# Extract the wound segment area from the input image
wound_segment = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

# Get the dimensions of the background image
bg_height, bg_width, _ = background_image.shape

# Resize the wound segment to match the background image dimensions
wound_segment = cv2.resize(wound_segment, (bg_width, bg_height))

# Create a mask for the wound segment
wound_mask = cv2.bitwise_not(binary_mask)
wound_mask = cv2.resize(wound_mask, (bg_width, bg_height))

# Create a copy of the background image
output_image = background_image.copy()

# Superimpose the wound segment onto the background image
masked_wound = cv2.bitwise_and(wound_segment, wound_segment, mask=wound_mask)
output_image = cv2.bitwise_or(output_image, masked_wound)

# Save the output image
output_file_path = f"{output_dir}/superimposed_wound.jpg"
cv2.imwrite(output_file_path, output_image)
print(f"Output image saved to: {output_file_path}")
