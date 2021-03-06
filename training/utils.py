import tensorflow as tf
import os
import errno
from glob import glob

import cv2
import numpy as np
import time
from tensorflow.keras.preprocessing import image


JUSTRGB = False
EXPANDED_LABELS = True


@tf.function
def dataset_length(data_dir):
    if os.path.isdir(data_dir):
        input_files = tf.io.gfile.glob(os.path.join(data_dir, "*"))
        data_set = tf.data.TFRecordDataset(input_files)
    elif os.path.isfile(data_dir):
        data_set = tf.data.TFRecordDataset(data_dir)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)

    return sum(1 for record in iter(data_set))


def get_dataset(
    filename, batch_size, justRGB=False, expanded=False, ca_flag=False, simclr=False
):
    if os.path.isdir(filename):
        filename = [f for f in glob(os.path.join(filename, "*.tfrecord"))]
    elif not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    return get_batched_dataset(
        filename,
        batch_size,
        justRGB,
        expanded_labels=expanded,
        ca=ca_flag,
        simclr=simclr,
    )


def _get_binary_label(example):
    if not EXPANDED_LABELS:
        return example["original_labels_multi_hot"][tf.constant(12)]
    return (
        example["original_labels_multi_hot"][tf.constant(12)]
        | example["original_labels_multi_hot"][tf.constant(13)]
        | example["original_labels_multi_hot"][tf.constant(14)]
        | example["original_labels_multi_hot"][tf.constant(15)]
        | example["original_labels_multi_hot"][tf.constant(16)]
    )


def read_tfrecord(example):
    """
    THIS FUNCTION IS USED TO PARSE THE TFRECORDS FILES FOR BIGEARTHNET DATA.
    THE BAND STATISTICS WERE PROVIDED BY THE BIGEARTHNET TEAM
    """
    BAND_STATS = {
        "mean": {
            "B01": 340.76769064,
            "B02": 429.9430203,
            "B03": 614.21682446,
            "B04": 590.23569706,
            "B05": 950.68368468,
            "B06": 1792.46290469,
            "B07": 2075.46795189,
            "B08": 2218.94553375,
            "B8A": 2266.46036911,
            "B09": 2246.0605464,
            "B11": 1594.42694882,
            "B12": 1009.32729131,
        },
        "std": {
            "B01": 554.81258967,
            "B02": 572.41639287,
            "B03": 582.87945694,
            "B04": 675.88746967,
            "B05": 729.89827633,
            "B06": 1096.01480586,
            "B07": 1273.45393088,
            "B08": 1365.45589904,
            "B8A": 1356.13789355,
            "B09": 1302.3292881,
            "B11": 1079.19066363,
            "B12": 818.86747235,
        },
    }

    # Use this one-liner to standardize each feature prior to reshaping.
    def standardize_feature(data, band_name):
        """
        APPLY STANDARDIZATION AND SCALING CONSISTENT WITH BEN PROCEDURE
        """
        return (
            tf.dtypes.cast(data, tf.float32) - BAND_STATS["mean"][band_name]
        ) / BAND_STATS["std"][band_name]

    # decode the TFRecord
    # The parse single example methods takes an example (from a tfrecords file),
    # and a dictionary that explains the data format of each feature.
    example = tf.io.parse_single_example(
        example,
        {
            "B01": tf.io.FixedLenFeature([20 * 20], tf.int64),
            "B02": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B03": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B04": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B05": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B06": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B07": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B08": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B8A": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B09": tf.io.FixedLenFeature([20 * 20], tf.int64),
            "B11": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "B12": tf.io.FixedLenFeature([60 * 60], tf.int64),
            "patch_name": tf.io.VarLenFeature(dtype=tf.string),
            "original_labels": tf.io.VarLenFeature(dtype=tf.string),
            "original_labels_multi_hot": tf.io.FixedLenFeature([43], tf.int64),
        },
    )

    example["binary_label"] = _get_binary_label(example)

    # After parsing our data into a tensor, let's standardize and reshape.
    reshaped_example = {
        "B01": tf.reshape(standardize_feature(example["B01"], "B01"), [20, 20]),
        "B02": tf.reshape(standardize_feature(example["B02"], "B02"), [120, 120]),
        "B03": tf.reshape(standardize_feature(example["B03"], "B03"), [120, 120]),
        "B04": tf.reshape(standardize_feature(example["B04"], "B04"), [120, 120]),
        "B05": tf.reshape(standardize_feature(example["B05"], "B05"), [60, 60]),
        "B06": tf.reshape(standardize_feature(example["B06"], "B06"), [60, 60]),
        "B07": tf.reshape(standardize_feature(example["B07"], "B07"), [60, 60]),
        "B08": tf.reshape(standardize_feature(example["B08"], "B08"), [120, 120]),
        "B8A": tf.reshape(standardize_feature(example["B8A"], "B8A"), [60, 60]),
        "B09": tf.reshape(standardize_feature(example["B09"], "B09"), [20, 20]),
        "B11": tf.reshape(standardize_feature(example["B11"], "B11"), [60, 60]),
        "B12": tf.reshape(standardize_feature(example["B12"], "B12"), [60, 60]),
        "patch_name": example["patch_name"],
        "original_labels": example["original_labels"],
        "original_labels_multi_hot": example["original_labels_multi_hot"],
        "binary_labels": example["binary_label"],
    }

    # Next sort the layers by resolution
    bands_10m = tf.stack(
        [
            reshaped_example["B04"],
            reshaped_example["B03"],
            reshaped_example["B02"],
            reshaped_example["B08"],
        ],
        axis=2,
    )

    bands_20m = tf.stack(
        [
            reshaped_example["B05"],
            reshaped_example["B06"],
            reshaped_example["B07"],
            reshaped_example["B8A"],
            reshaped_example["B11"],
            reshaped_example["B12"],
        ],
        axis=2,
    )

    # SG: for ImageNet pretrained models we can just pass the three RGB channels
    if JUSTRGB:
        img = tf.stack(
            [
                reshaped_example["B04"],
                reshaped_example["B03"],
                reshaped_example["B02"],
            ],
            axis=2,
        )
    else:
        # Finally resize the 20m data and stack the bands together.
        img = tf.concat(
            [bands_10m, tf.image.resize(bands_20m, [120, 120], method="bicubic")],
            axis=2,
        )

    multi_hot_label = reshaped_example["original_labels_multi_hot"]
    binary_label = reshaped_example["binary_labels"]

    # Can update this to return the multilabel if doing multi-class classification
    return img, binary_label


def read_ca_tfrecord(example):
    """
    THE CALIFORNIA DATA HAS DIFFERENT POPULATION STATISTICS AS EXPECETED.
    CALCULATED VIA THE process_california_data.ipynb file
    """
    BAND_STATS = {
        "mean": {
            "B02": 745.8342280288207,
            "B03": 1066.1362867829712,
            "B04": 1294.678473044234,
            "B05": 1645.7598649250806,
            "B06": 2246.824426424665,
            "B07": 2516.3336991935817,
            "B08": 2688.8463771950937,
            "B8A": 2733.816949232295,
            "B11": 2769.942382613557,
            "B12": 2092.625560070325,
        },
        "std": {
            "B02": 504.9172431483328,
            "B03": 616.4692423335321,
            "B04": 851.3811496920607,
            "B05": 795.0872173538605,
            "B06": 765.746057996193,
            "B07": 871.266391942569,
            "B08": 919.4293720949656,
            "B8A": 891.7677760562052,
            "B11": 1083.5092422778923,
            "B12": 1101.34386721669,
        },
    }

    # Use this one-liner to standardize each feature prior to reshaping.
    def standardize_feature(data, band_name):
        return (
            tf.dtypes.cast(data, tf.float32) - BAND_STATS["mean"][band_name]
        ) / BAND_STATS["std"][band_name]

    # decode the TFRecord
    # The parse single example methods takes an example (from a tfrecords file),
    # and a dictionary that explains the data format of each feature.
    example = tf.io.parse_single_example(
        example,
        {
            "B02": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B03": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B04": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B05": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B06": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B07": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B08": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B8A": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B11": tf.io.FixedLenFeature([120 * 120], tf.int64),
            "B12": tf.io.FixedLenFeature([120 * 120], tf.int64),
        },
    )

    # After parsing our data into a tensor, let's standardize and reshape.
    reshaped_example = {
        "B02": tf.reshape(standardize_feature(example["B02"], "B02"), [120, 120]),
        "B03": tf.reshape(standardize_feature(example["B03"], "B03"), [120, 120]),
        "B04": tf.reshape(standardize_feature(example["B04"], "B04"), [120, 120]),
        "B05": tf.reshape(standardize_feature(example["B05"], "B05"), [120, 120]),
        "B06": tf.reshape(standardize_feature(example["B06"], "B06"), [120, 120]),
        "B07": tf.reshape(standardize_feature(example["B07"], "B07"), [120, 120]),
        "B08": tf.reshape(standardize_feature(example["B08"], "B08"), [120, 120]),
        "B8A": tf.reshape(standardize_feature(example["B8A"], "B8A"), [120, 120]),
        "B11": tf.reshape(standardize_feature(example["B11"], "B11"), [120, 120]),
        "B12": tf.reshape(standardize_feature(example["B12"], "B12"), [120, 120]),
    }

    # Next sort the layers by resolution - all the same resolution for CA
    bands_10m = tf.stack(
        [
            reshaped_example["B04"],
            reshaped_example["B03"],
            reshaped_example["B02"],
            reshaped_example["B08"],
        ],
        axis=2,
    )

    bands_20m = tf.stack(
        [
            reshaped_example["B05"],
            reshaped_example["B06"],
            reshaped_example["B07"],
            reshaped_example["B8A"],
            reshaped_example["B11"],
            reshaped_example["B12"],
        ],
        axis=2,
    )

    # Finally resize the 20m data and stack the bands together.
    img = tf.concat([bands_10m, bands_20m], axis=2)

    return img, 0


def get_batched_dataset(
    filenames,
    batch_size,
    justRGB=False,
    augment=False,
    simclr=False,
    ca=False,
    expanded_labels=True,
):
    """
    This function is used to return a batch generator for training our tensorflow model.
    basically we read from different tfrecords files, and shuffle our records.
    we use the appropriate parsing function depending on if it is CA data or BigEarthNet data
    Finally - if it is a SimCLR model do not repeat the dataset, as we manually loop over our data
    and train our model in the simclr.py script.
    """
    global JUSTRGB, EXPANDED_LABELS
    JUSTRGB = justRGB
    EXPANDED_LABELS = expanded_labels

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames, shuffle=True)
    # print(f"Filenames: {filenames}")
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=2, num_parallel_calls=1
    )

    if simclr:
        dataset = dataset.shuffle(buffer_size=2048)
    else:
        dataset = dataset.shuffle(buffer_size=2048).repeat()

    if ca:
        dataset = dataset.map(read_ca_tfrecord, num_parallel_calls=10)
    else:
        dataset = dataset.map(read_tfrecord, num_parallel_calls=10)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(5)  #

    return dataset


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class Augment:
    # Default augmentations are on, can be turned on individually if parameters provided
    def __init__(self, augmentations):
        self.blur = "blur" in augmentations
        self.brightness = "brightness" in augmentations
        self.contrast = "contrast" in augmentations
        self.gain = "gain" in augmentations

        self.scale = 0.5

    def augfunc(self, sample):
        # Randomly apply transformation (color distortions) with probability p.
        if self.brightness:
            sample = self._random_apply(self._brightness, sample, p=0.8)

        if self.contrast:
            sample = self._random_apply(self._contrast, sample, p=0.8)

        if self.gain:
            sample = self._random_apply(self._gain, sample, p=0.8)

        if self.blur:
            sample = self._random_apply(self._blur, sample, p=0.8)

        return sample

    def _brightness(self, x):
        return tf.image.random_brightness(x, max_delta=0.8 * self.scale)

    def _contrast(self, x):
        return tf.image.random_contrast(
            x, lower=1 - 0.8 * self.scale, upper=1 + 0.8 * self.scale
        )

    def _gain(self, x):
        g = np.random.uniform(-self.scale, self.scale)
        return tf.image.adjust_gamma(x, gamma=1.0, gain=g)

    def _blur(self, x):
        # SimClr implementation is applied at 10% of image size with a random sigma
        p = np.random.uniform(0.1, 2)
        if type(x) == np.ndarray:
            return cv2.GaussianBlur(x, (5, 5), p)
        return cv2.GaussianBlur(x.numpy(), (5, 5), p)

    def _random_apply(self, func, x, p):
        return tf.cond(
            tf.less(
                tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32),
            ),
            lambda: func(x),
            lambda: x,
        )
