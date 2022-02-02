import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV2, NASNetMobile
import matplotlib.pyplot as plt

import mlflow

from config import PHOTOS_DIR, MODELS_DIR, MODEL_LOG, LOGS

FIG_DIR = LOGS / 'figures'

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
    fig_size = (sub_plots * 2) + 4 / 3 * sub_plots
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


def mobilenet_transfer_learning(mobel_net):
    base_model = mobel_net(weights='imagenet',
                           input_shape=(*mobile_net_image_size, 3),
                           include_top=False)
    base_model.trainable = False

    dropout_rate = .2

    inputs = tf.keras.Input(shape=(*original_image_size, 3))
    x = layers.experimental.preprocessing.Resizing(*resize_image_size)(inputs)
    x = layers.Lambda(lambda xx: tf.image.resize_with_pad(xx, *mobile_net_image_size,
                                                          method='bilinear',
                                                          antialias=False))(x)
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  metrics=['accuracy'])

    initial_epochs = 10

    model_path = MODELS_DIR / 'mobilenetV2_fine_tuned.h5'
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=4)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                             save_best_only=True)
    run_index = 6  # increment every time you train the model
    run_logdir = MODEL_LOG / f'mobile_net_{run_index}'
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    callbacks = [early_stopping_cb,
                 model_checkpoint_cb,
                 tensorboard_cb
                 ]

    history_first_pass = model.fit(train_ds, epochs=initial_epochs, class_weight=class_weights,
                                   validation_data=val_ds, callbacks=callbacks)
    # only track the fine-tuning
    with mlflow.start_run():
        mlflow.log_param("image_size", resize_image_size)
        mlflow.log_param("mobile_net_imsize", mobile_net_image_size)
        mlflow.log_param("dropout", dropout_rate)
        base_model.trainable = True

        fine_tune_at = 0
        mlflow.log_param("layers_remain_frozen", fine_tune_at)

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
                      metrics=['accuracy'])

        fine_tune_epochs = 10
        total_epochs = initial_epochs + fine_tune_epochs

        history_fine_tune = model.fit(train_ds, epochs=total_epochs, class_weight=class_weights,
                                      initial_epoch=history_first_pass.epoch[-1],
                                      validation_data=val_ds, callbacks=callbacks)
        accuracy = history_first_pass.history['accuracy']
        val_acc = history_first_pass.history['val_accuracy']

        loss = history_first_pass.history['loss']
        val_loss = history_first_pass.history['val_loss']

        accuracy += history_fine_tune.history['accuracy']
        val_acc += history_fine_tune.history['val_accuracy']

        loss += history_fine_tune.history['loss']
        val_loss += history_fine_tune.history['val_loss']
        plot_history(accuracy, val_acc, loss, val_loss, initial_epochs)

        return history_fine_tune, model


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


def plot_history(accuracy, val_accuracy, loss, val_loss, initial_epochs):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(FIG_DIR / "history.png")
    mlflow.log_artifact(FIG_DIR / "history.png")


if __name__ == '__main__':
    # hist, mod = train_cnn()
    hist, mode = mobilenet_transfer_learning(MobileNetV2)
