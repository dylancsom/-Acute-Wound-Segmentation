import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils import register_keras_serializable
from keras import backend as K
import cv2
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode


def open_webcam():
    js = Javascript('''
        async function openWebcam() {
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            return video;
        }
    ''')
    display(js)
    video = eval_js('openWebcam()')
    return video

# ... (keep the existing function definitions and model loading)
background_ratio = 0.8
wound_ratio = 0.2
class_weights = {0: 1.0, 1: background_ratio / wound_ratio}

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

if __name__ == '__main__':
    smooth = 10e-6
    y_pred = np.zeros((1, 128, 128))
    # one pixel is set to 1
    y_pred[0, 0, 0] = 1
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.zeros((1, 128, 128), dtype=tf.float32)
    print(dice_coef(y_true, y_pred, smooth=smooth))
    print(dice_coef_loss(y_true, y_pred, smooth=smooth))

# Load the trained model
model_path = '/content/drive/MyDrive/model.v3.10/trials/june11v9.keras'
model = tf.keras.models.load_model(model_path)

# Open the video capture
video = open_webcam()

while True:
    # Capture a new frame from the webcam
    frame = eval_js('video.currentFrame.data')
    frame = np.frombuffer(frame, dtype=np.uint8).reshape((video.videoHeight, video.videoWidth, 4))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Preprocess the frame
    frame = cv2.resize(frame, (256, 256))
    frame = img_to_array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Make predictions on the frame
    prediction = model.predict(frame)

    # Convert the prediction to a binary mask
    binary_mask = (prediction[0, :, :, 0] > 0.3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find the contours of the predicted wound area
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the frame
    frame_with_contour = frame[0].copy() * 255
    frame_with_contour = cv2.drawContours(frame_with_contour.astype(np.uint8), contours, -1, (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Live Feed with Wound Outline', frame_with_contour)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cv2.destroyAllWindows()