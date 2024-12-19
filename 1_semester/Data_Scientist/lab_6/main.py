import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_DIR = "C:\\Users\\lutdi\\PycharmProjects\\bd_6_1\\images"
MASK_DIR = "C:\\Users\\lutdi\\PycharmProjects\\bd_6_1\\labels"
IMG_SIZE = (128, 128)

def load_data(image_dir, mask_dir):
    images = []
    masks = []
    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMG_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, IMG_SIZE)
            images.append(image / 255.0)
            masks.append((mask > 127).astype(np.float32)) 
    print(f"Loaded {len(images)} images and masks")
    return np.array(images), np.array(masks)

images, masks = load_data(IMAGE_DIR, MASK_DIR)
masks = masks[..., np.newaxis]  

data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

seed = 42
image_generator = image_datagen.flow(X_train, batch_size=16, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=seed)

def train_data_generator(image_generator, mask_generator):
    for (img, mask) in zip(image_generator, mask_generator):
        yield img, mask

train_gen = train_data_generator(image_generator, mask_generator)

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def improved_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# Метрика IoU
def iou_metric(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

def build_unet_model(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.3)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.3)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.5)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    u1 = UpSampling2D((2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c5)

    model = Model(inputs, outputs)
    return model

def build_dense_conv_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(np.prod(IMG_SIZE), activation='sigmoid'),
        tf.keras.layers.Reshape((IMG_SIZE[0], IMG_SIZE[1], 1))
    ])
    return model

unet_model = build_unet_model(X_train.shape[1:])
unet_model.compile(optimizer=Adam(learning_rate=0.0001), loss=improved_loss, metrics=['accuracy', iou_metric])

dense_conv_model = build_dense_conv_model(X_train.shape[1:])
dense_conv_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

unet_history = unet_model.fit(
    train_gen,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping],
    steps_per_epoch=len(X_train) // 16
)

dense_conv_history = dense_conv_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping],
    batch_size=16
)

unet_pred_masks = unet_model.predict(X_test)
unet_pred_masks = (unet_pred_masks > 0.5).astype(np.uint8)

dense_pred_masks = dense_conv_model.predict(X_test)
dense_pred_masks = (dense_pred_masks > 0.5).astype(np.uint8)

print("Unique values in U-Net predictions:", np.unique(unet_pred_masks))
print("Unique values in Dense predictions:", np.unique(dense_pred_masks))

unet_accuracy = accuracy_score(y_test.flatten(), unet_pred_masks.flatten())
dense_accuracy = accuracy_score(y_test.flatten(), dense_pred_masks.flatten())

unet_precision = precision_score(y_test.flatten(), unet_pred_masks.flatten(), zero_division=1)
dense_precision = precision_score(y_test.flatten(), dense_pred_masks.flatten(), zero_division=1)

unet_recall = recall_score(y_test.flatten(), unet_pred_masks.flatten(), zero_division=1)
dense_recall = recall_score(y_test.flatten(), dense_pred_masks.flatten(), zero_division=1)

unet_f1 = f1_score(y_test.flatten(), unet_pred_masks.flatten(), zero_division=1)
dense_f1 = f1_score(y_test.flatten(), dense_pred_masks.flatten(), zero_division=1)

print("U-Net Metrics:")
print(f"Accuracy: {unet_accuracy}, Precision: {unet_precision}, Recall: {unet_recall}, F1-Score: {unet_f1}")

print("Dense Conv Model Metrics:")
print(f"Accuracy: {dense_accuracy}, Precision: {dense_precision}, Recall: {dense_recall}, F1-Score: {dense_f1}")

def visualize_results(images, true_masks, pred_masks, title):
    plt.figure(figsize=(15, 10))
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Исходное изображение")
        plt.axis('off')

        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(true_masks[i].squeeze(), cmap='gray')
        plt.title("Исходная маска")
        plt.axis('off')

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(pred_masks[i].squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

visualize_results(X_test, y_test, unet_pred_masks, "U-Net Предсказанная маска")
visualize_results(X_test, y_test, dense_pred_masks, "Dense Conv Предсказанная маска")
