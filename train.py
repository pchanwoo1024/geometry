import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

data_dir = pathlib.Path("dataset")
img_size = (224,224)
batch_size = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    seed=42, image_size=img_size, batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    seed=42, image_size=img_size, batch_size=batch_size)

base = tf.keras.applications.MobileNetV2(
    input_shape=img_size+(3,), include_top=False, weights="imagenet")
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(8, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("snack_classifier.h5")
