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
from tensorflow.keras.applications import ResNet50, ResNet101V2, ResNet152, Xception, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from utils import *

print(f"Using TensorFlow Version: {tf.__version__}")

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



def build_model(student_model, metrics=METRICS, dropout=0.25) :
    #define our student learner model
    model = student_model(
                include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=[120, 120, 10],
                pooling=None,)
    model.trainable = True

    # add new classifier layers
    flat = tf.keras.layers.Flatten()(model.layers[-1].output)
    h1 = tf.keras.layers.Dense(1024, activation="elu")(flat)
    h1 = tf.keras.layers.Dropout(dropout)(h1)
    h2 = tf.keras.layers.Dense(512, activation="elu")(h1)
    h2 = tf.keras.layers.Dropout(dropout)(h2)
    clf = tf.keras.layers.Dense(256, activation="elu")(h2)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(clf)
    # define new model
    model = tf.keras.models.Model(inputs=model.inputs, outputs=output)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=metrics)

    return model


def run_model(teacher_model_path, architecture, training_dataset, args) :

    # Loads previously trained model, set untrainable to make it clear it's only for prediction
    teacher_model = tf.keras.models.load_model(teacher_model_path)
    teacher_model.trainable = False  

    # Build our student model
    student = build_model(architecture)

    training_data = get_dataset(training_dataset, batch_size=args.BATCH_SIZE, expanded=args.expanded_labels)
    len_train_records = dataset_length(training_dataset)
    steps_per_epoch = len_train_records // args.BATCH_SIZE

    print("Length: ",len_train_records,", Steps:",steps_per_epoch)
    batch_generator = iter(training_data)

    # Manually walk through epochs and batches
    for epoch in tqdm(range(args.EPOCHS)) :

        # Loop over batches
        for step in tqdm(range(steps_per_epoch)):
            batch = batch_generator.get_next()
            # Get teacher prediction on batches labels
            distill_labels = np.squeeze(np.rint(teacher_model.predict(batch[0]))).astype("int")

            stats = student.train_on_batch(batch[0], distill_labels, return_dict=True,)

            # TODO store stats for saving to file later?
            #print(step, distill_labels,"Stats:", stats)

        # TODO add validation stat?

    # Eval student/teacher models on Test dataset
    student_stats = student.evaluate(training_data, steps=steps_per_epoch, return_dict=True)
    df = pd.DataFrame.from_dict(student_stats, orient="index", columns=["student"])
    print("student_test_stats",student_stats)

    teacher_stats = teacher_model.evaluate(training_data, steps=steps_per_epoch, return_dict=True)
    df2 = pd.DataFrame.from_dict(teacher_stats, orient="index", columns=["teacher"])
    print("teacher_test_stats",teacher_stats)

    df_all = pd.concat([df, df2], axis=1)
    # Save student out to disk
    student.save(f'{args.output_dir}/{args.output}.h5')
    df_all.to_pickle(f'{args.output_dir}/{args.output}.pkl')

    print(df_all)


if __name__ == '__main__':
    
    print('In main function')
    parser = argparse.ArgumentParser(
        description="Script for distill training on several student classifiers"
    )
    parser.add_argument('-m', '--model', required=True,
                        help='Which pretrained model do you want to distill from?')
    parser.add_argument(
        "-a",
        "--arch",
        choices=["ResNet50", "ResNet101V2", "ResNet152", "Xception", "InceptionV3"],
        default="ResNet50",
        help="Class of Model Architecture to use for student",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str, required=True,
        help="Output File Prefix for model file and dataframe",
    )
    parser.add_argument(
        "-f",
        "--output-dir",
        type=str, required=True,
        help="Output directory for model file and dataframe",
    )
    parser.add_argument(
        "-b",
        "--BATCH_SIZE",
        default=32,
        type=int,
        help="batch size to use during training",
    )
    parser.add_argument(
        "-e", "--EPOCHS", default=50, type=int, help="number of epochs to run"
    )
    parser.add_argument('-d', '--train_data', required=True, type=str,
                    help="Folder or filepath to tf records for training.")
    parser.add_argument("--expanded-labels",
                        action="store_true",
                        help="Whether to use expanded irrigation labels",)


    args = parser.parse_args()

    arch_dict = {'ResNet50': ResNet50,
                 'ResNet101V2':ResNet101V2,
                 'ResNet152': ResNet152,
                 'Xception':Xception,
                 'InceptionV3':InceptionV3}

    if args :

        run_model(
                args.model,
                arch_dict[args.arch],
                args.train_data,
                args)