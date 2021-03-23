# This is a test script to check what has been pickled for the image patch files
ROOT='/home/cagastya/hdd/gee_central_valley/Chico_S2SR_9_2019_39.625_-121.875'
FILENAME = 'Chico_S2SR_9_2019_39.625_-121.875.pkl.bad'
RP1 = '/hdd/BigEarthNet-v1.0/S2A_MSIL2A_20170717T113321_28_87'
FN1 = 'S2A_MSIL2A_20170717T113321_28_87_B06.tif'

import pickle
import rasterio
import numpy as np
import os
import tensorflow as tf
import errno
from glob import glob

def read_pickle_file():
    with open (os.path.join(ROOT, FILENAME), 'rb') as fp:
        data = pickle.load(fp)

        print(f'Dictionary has rows: {len(data)}')

        for idx, key in enumerate(data):
            print(f'{data[key]}')
            if idx == 0:
                break

# @tf.function
def dataset_length(data_dir):
    if os.path.isdir(data_dir):
        input_files = tf.io.gfile.glob(os.path.join(data_dir, "*"))
        data_set = tf.data.TFRecordDataset(input_files)
    elif os.path.isfile(data_dir):
        data_set = tf.data.TFRecordDataset(data_dir)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)

    return sum(1 for record in iter(data_set))

def read_from_image():
    band_ds = rasterio.open(path.join(RP1, FN1))
    band_data = np.array(band_ds.read(1))
    print(band_data)

if __name__ == "__main__":
    print('\n---From GEE---------------------')
    read_pickle_file()

    len_clean_records = dataset_length(os.path.join('/home/cagastya/MIDS_Capstone/data/Tfrecords.CentralValley',
        'CV_sample.tfrecord'))
    print(f"Clean records: {len_clean_records}")

    #print('\n---From BEN---------------------')
    #read_from_image()
