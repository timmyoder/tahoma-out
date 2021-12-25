import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from config import PHOTOS_DIR

image_size = (256, 256)
batch_size = 32


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PHOTOS_DIR,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PHOTOS_DIR,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")
#
#
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
#
#
# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)
#     # Image augmentation block
#     # Entry block
#     x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     x = layers.Conv2D(64, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     previous_block_activation = x  # Set aside residual
#
#     for size in [128, 256, 512, 728]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#         # Project residual
#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     x = layers.GlobalAveragePooling2D()(x)
#     if num_classes == 2:
#         activation = "sigmoid"
#         units = 1
#     else:
#         activation = "softmax"
#         units = num_classes
#
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(units, activation=activation)(x)
#     return keras.Model(inputs, outputs)
#
#
# model = make_model(input_shape=image_size, num_classes=4)
# keras.utils.plot_model(model, show_shapes=True)
#
# epochs = 50
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss="binary_crossentropy",
#     metrics=["accuracy"],
# )
# model.fit(
#     train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
# )

model = tf.keras.models.Sequential()
model.add(layers.Flatten(input_shape=(256, 256, 3)))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=20, validation_data=val_ds)


