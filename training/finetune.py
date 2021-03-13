import pandas as pd
import tensorflow as tf
from glob import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
import json
import time
from tensorflow.keras.layers import *

from utils import *
import helpers
import losses
import argparse
import cv2
from pprint import pprint


# Use the following metrics for evaluation
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


def load_pretrained_model(model, metrics=METRICS, hidden1=256, hidden2=256):

    pretrained_model = tf.keras.models.load_model(model)
    pretrained_model.trainable = True

    h1 = tf.keras.layers.Dense(hidden1, activation="elu", name="dense_ft_1")(
        pretrained_model.layers[-2].output
    )
    h1 = tf.keras.layers.Dropout(0.50, name="dropout_ft_1")(h1)
    h2 = tf.keras.layers.Dense(hidden2, activation="elu", name="dense_ft_2")(h1)
    h2 = tf.keras.layers.Dropout(0.50, name="dropout_ft_2")(h2)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(h2)

    # define new model
    new_model = tf.keras.models.Model(inputs=pretrained_model.inputs, outputs=output)

    # Learning rate of 5e-5 used for finetuning based on hyperparameter evaluations
    ft_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    # Compile model with Cross Entropy loss
    new_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=ft_optimizer,
        metrics=metrics,
    )

    return new_model


def finetune_pretrained_model(model, num_unfrozen, metrics=METRICS):
    """
    This function is used to simply finetune from the existing projection head, as opposed
    to stacking a new MLP on top of a projection head output as is done above.
    """
    pretrained_model = tf.keras.models.load_model(model)

    # Freeze all layers
    pretrained_model.trainable = False

    # Unfreeze just the projection head
    for layer in pretrained_model.layers[-num_unfrozen:]:
        layer.trainable = True

    # Add output layer
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(
        pretrained_model.layers[-1].output
    )

    # define new model
    new_model = tf.keras.models.Model(inputs=pretrained_model.inputs, outputs=output)

    ft_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    # Compile model with Cross Entropy loss
    new_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=ft_optimizer,
        metrics=metrics,
    )

    return new_model


def run_model(
    name,
    output_dir,
    pretrained_model,
    BATCH_SIZE,
    epochs,
    training_dataset,
    valid_dataset,
    test_dataset,
    NUM_UNFROZEN,
    expanded_labels=False,
):
    print(50 * "*")
    print(f"Running model: {name}")
    print(50 * "=")
    print(f"Batch Size: {BATCH_SIZE}")

    print("loading training data")
    training_data = get_dataset(
        training_dataset, batch_size=BATCH_SIZE, expanded=expanded_labels
    )
    print("loading validation data")
    val_data = get_dataset(
        valid_dataset, batch_size=BATCH_SIZE, expanded=expanded_labels
    )
    print("loading test data")
    test_data = get_dataset(
        test_dataset, batch_size=BATCH_SIZE, expanded=expanded_labels
    )

    len_val_records = dataset_length(valid_dataset)
    len_train_records = dataset_length(training_dataset)
    len_test_records = dataset_length(test_dataset)

    steps_per_epoch = len_train_records // BATCH_SIZE
    validation_steps = len_val_records // BATCH_SIZE
    test_steps = len_test_records // BATCH_SIZE

    print(f"Using {training_dataset} as training data.")
    print(f"{len_train_records} total records and {steps_per_epoch} steps per epoch")

    # Use an early stopping callback and our timing callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", verbose=1, patience=25, mode="max", restore_best_weights=True
    )

    time_callback = TimeHistory()

    print(f"Using Pretrained Model: {pretrained_model}")
    if NUM_UNFROZEN:
        print(
            f"Finetuning the SimCLR model from the {(NUM_UNFROZEN+1)//2} layer of the projection head"
        )
        model = finetune_pretrained_model(pretrained_model, NUM_UNFROZEN)
    else:
        print(f"Adding new MLP to second layer of Projection head")
        model = load_pretrained_model(pretrained_model)
    model.summary()

    history = model.fit(
        training_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        callbacks=[time_callback, early_stop],
    )
    # times = time_callback.times
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
            "pretrained_model": pretrained_model,
            "score": score,
        }
        json.dump(params, fp)

    return df.auc.max()


if __name__ == "__main__":

    print("In main function")
    parser = argparse.ArgumentParser(
        description="Script for running different supervised classifiers"
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        help="Output directory for new models .h5 and .pkl files.",
    )
    parser.add_argument(
        "-m", "--model", help="Which pretrained model do you want to finetune?"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output File Prefix for model file and dataframe",
    )
    parser.add_argument(
        "-b",
        "--BATCH_SIZE",
        default=32,
        type=int,
        help="batch size to use during training and validation",
    )
    parser.add_argument(
        "-e", "--EPOCHS", default=10, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-u",
        "--UNFROZEN",
        default=None,
        type=int,
        help="Number of layers of PH to unfreeze during finetuning. If none, will add new MLP ontop of second PH layer",
        choices=[1, 3, 5],
    )
    parser.add_argument(
        "-t",
        "--training-data",
        type=str,
        help="Path to training data folder or .tfrecord files",
    )
    parser.add_argument(
        "-v",
        "--validation-data",
        type=str,
        help="Path to validation data folder or .tfrecord files",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data folder or .tfrecord files",
    )
    parser.add_argument(
        "--expanded-labels",
        action="store_true",
        help="Whether to use expanded irrigation labels",
    )

    args = parser.parse_args()

    print(f"Using TensorFlow Version: {tf.__version__}")

    if args:

        best_score = run_model(
            name=args.output,
            output_dir=args.output_dir,
            pretrained_model=args.model,
            BATCH_SIZE=args.BATCH_SIZE,
            epochs=args.EPOCHS,
            training_dataset=args.training_data,
            valid_dataset=args.validation_data,
            test_dataset=args.test_data,
            NUM_UNFROZEN=args.UNFROZEN,
            expanded_labels=args.expanded_labels,
        )

        print(f"Best Score: {best_score}")
