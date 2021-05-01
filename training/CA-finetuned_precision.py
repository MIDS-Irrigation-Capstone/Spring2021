from glob import glob
import argparse
import csv
import errno
import json
import os
import time
import pathlib

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

BAND_STATS = {
        'mean': {
            'B01': 340.76769064,
            'B02': 429.9430203,
            'B03': 614.21682446,
            'B04': 590.23569706,
            'B05': 950.68368468,
            'B06': 1792.46290469,
            'B07': 2075.46795189,
            'B08': 2218.94553375,
            'B8A': 2266.46036911,
            'B09': 2246.0605464,
            'B11': 1594.42694882,
            'B12': 1009.32729131
        },
        'std': {
            'B01': 554.81258967,
            'B02': 572.41639287,
            'B03': 582.87945694,
            'B04': 675.88746967,
            'B05': 729.89827633,
            'B06': 1096.01480586,
            'B07': 1273.45393088,
            'B08': 1365.45589904,
            'B8A': 1356.13789355,
            'B09': 1302.3292881,
            'B11': 1079.19066363,
            'B12': 818.86747235
        }
    }

BAND_STATS_CA = {"mean": {
            "B02": 704.0660306667106,
            "B03": 1013.6625595886348,
            "B04": 1177.3966978795684,
            "B05": 1559.4157583764888,
            "B06": 2271.823718038332,
            "B07": 2570.590654856275,
            "B08": 2729.77884789601,
            "B8A": 2782.5820922432317,
            "B11": 2633.2600006272755,
            "B12": 1960.1242074549111
        },
        "std": {
            "B02": 275.44633762623107,
            "B03": 341.9727574411451,
            "B04": 469.3295063736902,
            "B05": 439.3445385554108,
            "B06": 405.9869176625257,
            "B07": 462.80766566429446,
            "B08": 453.4091836811935,
            "B8A": 465.3067073795589,
            "B11": 570.6215399891606,
            "B12": 593.4589000748651
        }}


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


def run_model(teacher_model_path, training_dataset, output_dir, expanded, ca_flag) :
    SCALE_FACTOR = 4000 if ca_flag else 3000

    def denorm_img(img):
            
        if ca_flag:
            band_stats = BAND_STATS_CA
        else:
            band_stats = BAND_STATS
        
        return np.stack([(img[:,:,0]* band_stats['std']['B04']+ band_stats['mean']['B04'])/ SCALE_FACTOR,
                        (img[:,:,1]* band_stats['std']['B03']+ band_stats['mean']['B03'])/ SCALE_FACTOR,
                        (img[:,:,2]* band_stats['std']['B02']+ band_stats['mean']['B02'])/ SCALE_FACTOR], axis=2)
        
    def rgb_to_rgba32(img):
        """
        Convert an RGB image to a 32 bit-encoded RGBA image.
        """
        img = denorm_img(img)
        # Ensure it has three channels
        if len(img.shape) != 3 or img.shape[2] !=3:
            raise RuntimeError('Input image is not RGB.')

        # Get image shape
        n, m, _ = img.shape

        # Convert to 8-bit, which is expected for viewing
        im_8 = np.uint8(img*255)

        # Add the alpha channel, which is expected by Bokeh
        #im_rgba = np.dstack((im_8, 255*np.ones_like(im_8[:,:,0])))

        # Reshape into 32 bit. Must flip up/down for proper orientation
        #return np.flipud(im_rgba.view(dtype=np.int32).reshape(n, m))
        return np.flipud(im_8)

    # Loads previously trained model, set untrainable to make it clear it's only for prediction
    teacher_model = tf.keras.models.load_model(teacher_model_path)
    teacher_model.trainable = False  

    training_data = get_dataset(training_dataset, batch_size=1, expanded=expanded,
                                ca_flag=ca_flag, simclr=True) # test flags for top 100 CA
    batch_generator = iter(training_data)
    len_train_records = 50194 #dataset_length(training_dataset)


    # Store top predictions (>0.99) and their image
    top_100s = {}

    # Loop over batches
    for step in tqdm(range(len_train_records)):
        batch = batch_generator.get_next()[0]
        # Get teacher prediction on batches labels
        predicts = teacher_model.predict(batch)[0][0]
        #print("teacher_model.predict:",predicts, batch.shape)

        if predicts > 0.99:
            top_100s[(step,predicts)] = batch[0]

        # enforce cap of 100, drop lowest key
        if len(top_100s.keys()) > 100 :
            sorted_keys = sorted(top_100s.keys(), key=lambda x: x[1])
            top_100s.pop(sorted_keys[0])
        

    # loops thru 100 and save batches to disk
    print("Found good candidates:", len(top_100s.keys()), "saving to disk...")
    i = 0
    for k in sorted(top_100s.keys(), key=lambda x: x[1], reverse=True) :
        image_path = os.path.join(output_dir,f'{i:03}.png')
        #print(i,"saving:",image_path,k)
        img = rgb_to_rgba32(top_100s[k])
        cv2.imwrite(image_path, cv2.cvtColor( img, cv2.COLOR_RGB2BGR))
        i += 1


def main(model_dir, training_dataset, output_dir) :
    model_filepaths = glob(os.path.join(model_dir,"*.h5"))

    for model in model_filepaths :
        basename = os.path.basename(model)
        expanded = basename.startswith("expanded_")
        out_path = os.path.join(output_dir, os.path.splitext(basename)[0])

        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True) 
        print(basename, expanded,out_path)

        run_model(model, training_dataset, out_path, expanded, ca_flag=True)

        #break

if __name__ == '__main__':
    
    print('In main function')
    parser = argparse.ArgumentParser(
        description="Script to output images of highest confidence."
    )
    parser.add_argument('-m', '--model_dir', required=True,
                        help='Path to finetuned models')
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str, required=True,
        help="Output directory for model file and dataframe",
    )
    parser.add_argument('-d', '--train_data', required=True, type=str,
                    help="Folder or filepath to tf records for training.")

    args = parser.parse_args()


    if args :
        main(args.model_dir,
             args.train_data,
             args.output_dir)