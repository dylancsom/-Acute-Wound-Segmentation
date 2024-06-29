import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import numpy as np
import tensorflow as tf
from functools import partial
import os
import PIL
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope

def image_mask_generator(image_dir, mask_dir, batch_size, target_size=(256, 256), input_format=['jpg', 'jpeg'], mask_format='png'):

    from keras.preprocessing.image import img_to_array, load_img

    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(f'.{input_format}')]
    mask_filenames = {os.path.basename(f): os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + f'.{mask_format}') for f in image_filenames}
    image_paths = [os.path.join(image_dir, f) for f in image_filenames]

    total_files = len(image_filenames)
    loaded_files = 0  # Track how many files have been loaded

    while loaded_files < total_files:
        batch_images = []
        batch_masks = []
        for i in range(loaded_files, min(loaded_files + batch_size, total_files)):
            img_path = image_paths[i]
            mask_path = mask_filenames[os.path.basename(img_path)]

            img = load_img(img_path, target_size=target_size, color_mode='rgb')
            img = img_to_array(img) / 255.0  # Rescale to [0, 1]

            mask = load_img(mask_path, target_size=target_size, color_mode='grayscale' if mask_format == 'png' else 'rgb')
            mask = img_to_array(mask) / 255.0  # Rescale to [0, 1]

            batch_images.append(img)
            batch_masks.append(mask)

        loaded_files += len(batch_images)
        yield np.array(batch_images), np.array(batch_masks)

background_ratio = 0.8
wound_ratio = 0.2
class_weights = {0: 1.0, 1: background_ratio / wound_ratio}

def dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def weighted_dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.gather(K.constant(list(class_weights.values())), K.cast(y_true_f, dtype='int32'))
    intersection = K.sum(y_true_f * y_pred_f * weights)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

def weighted_dice_coef_loss(y_true, y_pred, smooth=10e-6):
    dice_coef = weighted_dice_coef(y_true, y_pred, smooth)
    loss = 1 - dice_coef
    return loss

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou

def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

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


batch_size = 6
target_size = (256, 256)

train_images_path = ''
train_masks_path = ''
val_images_path = ''
val_masks_path = ''

train_generator = partial(image_mask_generator, batch_size=batch_size, target_size=(256, 256), input_format=['jpg', 'jpeg'], mask_format='png')
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator(train_images_path, train_masks_path),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
    )
)
val_generator = partial(image_mask_generator, batch_size=batch_size, target_size=(256, 256), input_format=['jpg', 'jpeg'], mask_format='png')
val_dataset = tf.data.Dataset.from_generator(
    image_mask_generator,
    args=(val_images_path, val_masks_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)
    )
)

total_files = len(os.listdir(train_images_path))


def build_unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.5)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.5)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.5)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.5)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.5)(c3)

    # Decoder
    u4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.5)(c4)

    u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.5)(c5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)


    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Build the U-Net model
model = build_unet_model()

# Compile
reduce_lr = ReduceLROnPlateau(monitor='val_iou_loss', factor=0.5, patience=5, min_lr=1e-6)
with custom_object_scope({
    'weighted_dice_coef_loss': weighted_dice_coef_loss,
    'weighted_dice_coef': weighted_dice_coef,
    'dice_coef_loss': dice_coef_loss,
    'dice_coef': dice_coef,
    'combined_dice_iou_loss': combined_dice_iou_loss,
    'iou': iou
}):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.008),
                  loss=combined_dice_iou_loss,
                  metrics=['accuracy', dice_coef, weighted_dice_coef, dice_coef_loss, weighted_dice_coef_loss, iou, iou_loss, combined_dice_iou_loss])
epochs = 150
batch_size_num = 6

# Modified dataset paths
train_generator = partial(image_mask_generator, target_size=target_size, input_format='jpg', mask_format='png')

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator(train_images_path, train_masks_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, target_size[0], target_size[1], 1), dtype=tf.float32)
    )
).repeat(epochs)

val_generator = partial(image_mask_generator, target_size=target_size, input_format='jpg', mask_format='png')

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator(val_images_path, val_masks_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, target_size[0], target_size[1], 1), dtype=tf.float32)
    )
).repeat(epochs)


# Model summary
model.summary()

early_stopper = EarlyStopping(monitor='val_weighted_dice_coef', mode='max', patience=8, restore_best_weights=True)
# Fit the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    batch_size=batch_size_num,
    steps_per_epoch=41,
    validation_data=val_dataset,
    callbacks=[reduce_lr, early_stopper]
)

plt.figure(figsize=(16, 4))
# Plot accuracy
plt.subplot(1, 4, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 4, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot Dice coefficient
plt.subplot(1, 4, 3)
plt.plot(history.history['dice_coef'], label='Training Dice Coefficient')
plt.plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Training and Validation Dice Coefficient')

# Plot IoU
plt.subplot(1, 4, 4)
plt.plot(history.history['iou'], label='Training IoU')
plt.plot(history.history['val_iou'], label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('Training and Validation IoU')

plt.tight_layout()
plt.show()

# Save the model
model_path = ''
model.save(model_path)
