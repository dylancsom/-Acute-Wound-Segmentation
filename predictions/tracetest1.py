import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import register_keras_serializable
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

@register_keras_serializable()
def weighted_dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.gather(K.constant(list(class_weights.values())), K.cast(y_true_f, dtype='int32'))
    intersection = K.sum(y_true_f * y_pred_f * weights)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)
    return dice

@register_keras_serializable()
def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

@register_keras_serializable()
def weighted_dice_coef_loss(y_true, y_pred, smooth=10e-6):
    dice_coef = weighted_dice_coef(y_true, y_pred, smooth)
    loss = 1 - dice_coef
    return loss

@register_keras_serializable()
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou

@register_keras_serializable()
def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

@register_keras_serializable()
def combined_dice_iou_loss(y_true, y_pred, smooth=1e-6, dice_weight=0.5, iou_weight=0.5, wound_weight=0.75):
    unweighted_dice_loss = dice_coef_loss(y_true, y_pred, smooth)
    weighted_dice_loss = weighted_dice_coef_loss(y_true, y_pred, smooth)
    combined_dice_loss = wound_weight * weighted_dice_loss + (1 - wound_weight) * unweighted_dice_loss

    iou_loss_val = iou_loss(y_true, y_pred, smooth)

    combined_loss = dice_weight * combined_dice_loss + iou_weight * iou_loss_val
    return combined_loss

# Load the trained model
model_path = '/content/drive/MyDrive/model.v3.10/trials/june05v6.keras'
model = tf.keras.models.load_model(model_path)

# Load and preprocess the new image
new_image_path = '/content/drive/MyDrive/model.v3.10/predictions/freshimages/testimage4.jpg'
new_image = load_img(new_image_path, target_size=(256, 256))
new_image = img_to_array(new_image) / 255.0
new_image = np.expand_dims(new_image, axis=0)

# Make predictions on the new image
prediction = model.predict(new_image)

# Convert the prediction to a binary mask
binary_mask = (prediction[0, :, :, 0] > 0.3).astype(np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Find the contours of the predicted wound area
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for visualization
image_with_contour = new_image[0].copy()

# Draw the contours on the image
image_with_contour = cv2.drawContours(image_with_contour, contours, -1, (0, 255, 0), 2)

# Visualize the prediction
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(new_image[0])
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_with_contour)
plt.title('Predicted Mask with Contour')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(opened, cmap='gray')
plt.title('Predicted Mask (Opened)')
plt.axis('off')

plt.tight_layout()
plt.show()