from glob import glob
import argparse
import csv
import errno
import json
import os
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# import seaborn as sns
# from matplotlib.cm import get_cmap
from tensorflow.keras.applications import ResNet50, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from utils import *

print(f"Using TensorFlow Version: {tf.__version__}")
# sns.set()

# Set Paths
BASE_PATH = "./BigEarthData"
# OUTPUT_PATH = os.path.join(BASE_PATH, "models")
TFR_PATH = os.path.join(BASE_PATH, "tfrecords")


METRICS = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]


def dataset_length(data_dir):
    input_files = tf.io.gfile.glob(os.path.join(data_dir, "*"))
    data_set = tf.data.TFRecordDataset(input_files)
    return sum(1 for record in data_set)


def get_dataset(filename, batch_size, justRGB, expanded):
    if os.path.isdir(filename):
        filename = [f for f in glob(os.path.join(filename, "*.tfrecord"))]
    elif not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    return get_batched_dataset(filename, batch_size, justRGB, expanded)


def build_model(
    imported_model, use_pretrain, metrics=METRICS, output_bias=None, dropout=0.25
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    if use_pretrain:
        # SG: making pre-trained weights work
        model = imported_model(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=[120, 120, 3],
            pooling=None,
        )
        model.trainable = False
    else:
        model = imported_model(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=[120, 120, 10],
            pooling=None,
        )
        model.trainable = True
    # add new classifier layers
    flat = tf.keras.layers.Flatten()(model.layers[-1].output)
    h1 = tf.keras.layers.Dense(1024, activation="elu")(flat)
    h1 = tf.keras.layers.Dropout(dropout)(h1)
    h2 = tf.keras.layers.Dense(512, activation="elu")(h1)
    h2 = tf.keras.layers.Dropout(dropout)(h2)
    clf = tf.keras.layers.Dense(256, activation="elu")(h2)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias
    )(clf)
    # define new model
    model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer="adam", metrics=metrics
    )
    #   print(f'Trainable variables: {model.trainable_weights}')

    return model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def augfunc(sample):
    # Randomly apply transformation (color distortions) with probability p.
    sample = _random_apply(_color_jitter, sample, p=0.8)
    sample = _random_apply(_color_drop, sample, p=0.2)
    sample = _random_apply(_blur, sample, p=0.5)

    return sample


def _color_jitter(x, s=1):
    # one can also shuffle the order of following augmentations
    # each time they are applied.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    dx = tf.image.random_saturation(x[:, :, :3], lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    dx = tf.image.random_hue(dx, max_delta=0.2 * s)
    x = tf.concat([dx, x[:, :, 3:]], axis=2)
    x = tf.clip_by_value(x, 0, 1)
    return x


def _color_drop(x):
    dx = tf.image.rgb_to_grayscale(x[:, :, :3])
    dx = tf.tile(dx, [1, 1, 3])
    x = tf.concat([dx, x[:, :, 3:]], axis=2)
    return x


def _blur(x):
    # SimClr implementation is applied at 10% of image size with a random sigma
    p = np.random.uniform(0.1, 2)
    if type(x) == np.ndarray:
        return cv2.GaussianBlur(x, (5, 5), p)
    return cv2.GaussianBlur(x.numpy(), (5, 5), p)


def _random_apply(func, x, p):
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )


def run_model(
    output_dir,
    name,
    BATCH_SIZE=32,
    epochs=50,
    weights=False,
    architecture=ResNet50,
    pretrain=False,
    augment=False,
    training_filenames=f"{TFR_PATH}/train.tfrecord",
    validation_filenames=f"{TFR_PATH}/val.tfrecord",
    test_filenames=f"{TFR_PATH}/test.tfrecord",
    expanded_labels=True,
):
    print(50 * "*")
    print(f"Running model: {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")

    len_train_records = dataset_length(training_filenames)
    len_val_records = dataset_length(validation_filenames)
    len_test_records = dataset_length(test_filenames)
    print(f"Training records: {len_train_records}")
    print(f"Validation records: {len_val_records}")
    print(f"Test records: {len_test_records}")

    if weights:
        # neg = 38400 - 984
        # pos = 984
        # total = neg + pos
        neg_weight = 1  # (1 / neg) * (total) / 2.0
        pos_weight = 40  # (1 / pos) * (total) / 2.0
        class_weight = {0: neg_weight, 1: pos_weight}
        print(f"Using Class Weights: ")
        print("\tWeight for Negative Class: {:.2f}".format(neg_weight))
        print("\tWeight for Positive Class: {:.2f}".format(pos_weight))
    else:
        class_weight = None
        print("Not Using Weights")

    training_data = get_dataset(
        training_filenames,
        batch_size=BATCH_SIZE,
        justRGB=pretrain,
        expanded=expanded_labels,
    )
    val_data = get_dataset(
        validation_filenames,
        batch_size=BATCH_SIZE,
        justRGB=pretrain,
        expanded=expanded_labels,
    )
    test_data = get_dataset(
        test_filenames,
        batch_size=BATCH_SIZE,
        justRGB=pretrain,
        expanded=expanded_labels,
    )

    steps_per_epoch = len_train_records // BATCH_SIZE
    validation_steps = len_val_records // BATCH_SIZE
    test_steps = len_test_records // BATCH_SIZE

    # Use an early stopping callback and our timing callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=15, mode="max", restore_best_weights=True
    )

    time_callback = TimeHistory()

    print(f"Using Model Architecture: {architecture}")

    model = build_model(imported_model=architecture, use_pretrain=pretrain)
    # print(f'Trainable variables: {model.trainable_weights}')
    model.summary()

    augmentations = None
    if augment:
        datagen = image.ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.10,
            height_shift_range=0.10,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.20,
            preprocessing_function=augfunc,
        )
        aug_data = datagen.flow(train_X, train_y, batch_size=BATCH_SIZE, shuffle=True)

        history = model.fit(
            aug_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps,
            callbacks=[time_callback, early_stop],
            class_weight=class_weight,
        )
        times = time_callback.times
        df = pd.DataFrame(history.history)
        df["times"] = time_callback.times
        augmentations = datagen.__dict__

    else:
        history = model.fit(
            training_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps,
            callbacks=[time_callback, early_stop],
            class_weight=class_weight,
        )
        times = time_callback.times
        df = pd.DataFrame(history.history)
        df["times"] = time_callback.times

    df.to_pickle(f"{output_dir}/{name}.pkl")
    model.save(f"{output_dir}/{name}.h5")

    print("Evaluating final model against test data")
    score = model.evaluate(test_data, steps=test_steps, verbose=True)
    with open(f"{output_dir}/{name}.json", "w") as fp:
        params = {
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "weights": weights,
            "architecture": architecture.__name__,
            "pretrain": pretrain,
            "augment": augment,
            "class_weight": class_weight,
            "augmentations": augmentations,
            "score": score,
        }
        json.dump(params, fp)

    return df


if __name__ == "__main__":

    print("In main function")
    parser = argparse.ArgumentParser(
        description="Script for running different supervised classifiers"
    )
    parser.add_argument(
        "-a",
        "--arch",
        choices=["ResNet50", "ResNet101V2", "Xception", "InceptionV3"],
        default="ResNet50",
        help="Class of Model Architecture to use for classification",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output File Prefix for model file and dataframe",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model file and dataframe",
    )
    parser.add_argument(
        "-b",
        "--BATCH_SIZE",
        default=32,
        type=int,
        help="batch size to use during training and validation",
    )
    parser.add_argument(
        "-e", "--EPOCHS", default=50, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        action="store_true",
        help="whether to use ImageNet pretrained",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="False",
        type=str,
        help="whether to use weights",
        choices=["True", "False"],
    )
    parser.add_argument(
        "-g",
        "--augment",
        default="False",
        type=str,
        choices=["True", "False"],
        help="whether to augment the training data",
    )
    parser.add_argument(
        "--train-set",
        type=str,
        help="Path to tfrecords for training set",
    )
    parser.add_argument(
        "--validation-set",
        type=str,
        help="Path to tfrecords for validation set",
    )
    parser.add_argument(
        "--test-set",
        default="/data/test",
        type=str,
        help="Path to tfrecords for validation set",
    )
    parser.add_argument(
        "--expanded-labels",
        action="store_true",
        help="Whether to use expanded irrigation labels",
    )

    # second variable to ignore unknown args
    args, _ = parser.parse_known_args()

    arch_dict = {
        "ResNet50": ResNet50,
        "ResNet101V2": ResNet101V2,
        "Xception": Xception,
        "InceptionV3": InceptionV3,
    }

    AUGMENT = False
    if args.augment == "True":
        AUGMENT = True

    print(args)
    run_model(
        args.output_dir,
        args.output,
        BATCH_SIZE=args.BATCH_SIZE,
        epochs=args.EPOCHS,
        weights=args.weights == "True",
        architecture=arch_dict[args.arch],
        pretrain=args.pretrained,
        augment=AUGMENT,
        training_filenames=args.train_set,
        validation_filenames=args.validation_set,
        test_filenames=args.test_set,
        expanded_labels=args.expanded_labels,
    )
