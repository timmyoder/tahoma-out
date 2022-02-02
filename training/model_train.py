import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import mlflow

from config import PHOTOS_DIR, MODELS_DIR, MODEL_LOG

mlflow.tensorflow.autolog()

original_image_size = (563, 1000)
resize_image_size = (64, 128)
mobile_net_image_size = (128, 128)
batch_size = 16

class_weights = {0: 2.48,
                 1: 1.2,
                 2: 1.,
                 3: 1.2}

dataset_options = {'directory': PHOTOS_DIR,
                   'labels': 'inferred',
                   'label_mode': 'int',
                   'validation_split': 0.2,
                   'seed': 1337,
                   'batch_size': batch_size,
                   'image_size': original_image_size}

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    subset="training",
    **dataset_options
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    subset="validation",
    **dataset_options
)


def plot_rand_images(square_num_pics):
    sub_plots = np.sqrt(square_num_pics)
    fig_size = (sub_plots * 2) + 4/3 * sub_plots
    if sub_plots // 2 != 0:
        raise ValueError('Value must be a perfect square')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(fig_size, fig_size))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(sub_plots, sub_plots, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(int(labels[i]))
            plt.axis("off")


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def train_simple_sequential():
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

    return history, model


def train_cnn():
    with mlflow.start_run():
        mlflow.log_param("image_size", resize_image_size)
        model = tf.keras.models.Sequential([
            layers.experimental.preprocessing.Resizing(*resize_image_size),
            layers.experimental.preprocessing.Rescaling(1. / 255),
            layers.Conv2D(64, 7, strides=2, activation='relu',
                          padding='same', input_shape=[*resize_image_size, 3]),
            layers.MaxPool2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPool2D(2),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.MaxPool2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        model_path = MODELS_DIR / 'simple_cnn_weighted_small.h5'
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8)
        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                                 save_best_only=True)
        run_index = 5  # increment every time you train the model
        run_logdir = MODEL_LOG / f'simple_cnn_run_{run_index}'
        tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
        callbacks = [early_stopping_cb,
                     model_checkpoint_cb,
                     tensorboard_cb
                     ]

        history = model.fit(train_ds, epochs=20, class_weight=class_weights,
                            validation_data=val_ds, callbacks=callbacks)

        return history, model


hist, mod = train_cnn()