import tensorflow as tf
from tensorflow.keras import layers

from config import PHOTOS_DIR, MODELS_DIR, MODEL_LOG

image_size = (256, 256)
batch_size = 32
dataset_options = {'directory': PHOTOS_DIR,
                   'labels': 'inferred',
                   'label_mode': 'int',
                   'validation_split': 0.2,
                   'seed': 1337,
                   'batch_size': batch_size}

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    subset="training",
    **dataset_options
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    subset="validation",
    **dataset_options
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

model = tf.keras.models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(*image_size))
model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model.add(layers.Flatten(input_shape=(*image_size, 3)))
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model_path = MODELS_DIR / 'simple_sequential_small.h5'
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8)
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                         save_best_only=True)
run_index = 2  # increment every time you train the model
run_logdir = MODEL_LOG / f'simple_sequential_run_{run_index}'
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb,
             model_checkpoint_cb,
             tensorboard_cb
             ]

history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
