import numpy as np
import os
import rasterio
import pickle
import tensorflow as tf
import traceback
import sys

# It appears that we are only interested in the 10nm and 20nm bands. So getting rid of B1 and B9
BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# According to Sentinel guide all MSI data is scaled by factor of 10000
SCALE_FACTOR = 1 #10000

def prep_example(ia_list, TFRecord_writer):
    try:
        progress_bar = tf.keras.utils.Progbar(target=len(ia_list))
        #bands, original_labels, original_labels_multi_hot, patch_name):
        for idx, key in enumerate(ia_list):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "B02": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B2"]))
                        ),
                        "B03": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B3"]))
                        ),
                        "B04": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B4"]))
                        ),
                        "B05": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B5"]))
                        ),
                        "B06": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B6"]))
                        ),
                        "B07": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B7"]))
                        ),
                        "B08": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B8"]))
                        ),
                        "B8A": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B8A"]))
                        ),
                        "B11": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B11"]))
                        ),
                        "B12": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=np.ravel(ia_list[key]["B12"]))
                        ),
                        "patch_name": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[ia_list[key]['patch_name'].encode("utf-8")])
                        ),
                        "month": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[str(ia_list[key]['month']).encode("utf-8")])
                        ),
                    }
                )
            )
            TFRecord_writer.write(example.SerializeToString())
            progress_bar.update(idx)
    except:
        # printing stack trace
        traceback.print_exception(*sys.exc_info())
        raise


def split_convert2TFrecord(TFRecord_writer,
    patch,
    month,
    base_path, band_names=BAND_NAMES,
    img_dim=120, scale = SCALE_FACTOR):
    '''
    base_path: String - Path to folder with tif file for each spectral band
    patch: name of the image folder
    band_names: List - List of bands to iterate over
    img_dim: int -  Dimension of final images desired - assumes square i.e., nxn
    scale: numeric - Value by which to divide the arrays of data in the tiff file

    '''

    # List to store temporary dataframes
    ia_list = {}

    try:
        print(f'Processing image patch {patch}')
        # Loop over all bands and read the actual MSI data in and create 3d array
        for b, band in enumerate(band_names):
            with rasterio.open(os.path.join(base_path, f'{patch}_msi_{band}.tif')) as data_ds:
                data = data_ds.read(1)

                # Split the large 3d array into many subpieces - first in the row direction
                rows = np.split(data, np.arange(img_dim, data.shape[0], img_dim))

                # Loop over the broken up rows and split up columns
                for c, col_chunk in enumerate(rows[:-1]):
                    img_arrays = np.split(col_chunk, np.arange(img_dim, data.shape[1], img_dim), axis=1)

                    # Store the small MSI and prediction list
                    for i, ia in enumerate(img_arrays[:-1]):
                        key = f'{c}_{i}'
                        if not key in ia_list:
                            ia_list[key] = {}
                            ia_list[key]['patch_name'] = base_path
                            ia_list[key]['month'] = month
                        #ia_list[key][band] = (np.int_(ia / scale))
                        ia_list[key][band] = (np.int_(ia))


        # Create TfRecords for the split msi
        prep_example(ia_list, TFRecord_writer)

    except Exception as e:
            print(f"Error: {str(e)}")

    #save the dictionary as a pickle file
    print(f'dumping the pkl file {patch}.pkl on disk')
    pickle.dump(ia_list, open(f'{base_path}/{patch}.pkl', 'wb'))

def get_month(patch):
    if patch:
        return int(patch.split('_')[2])
    return None

def create_split(root_folder,
    out_folder):

    #Create TFWriter for entire dataset
    try:
        print('Creating TFRecord writer')
        tfwriter = tf.io.TFRecordWriter(os.path.join(out_folder, "CV_sample.tfrecord"))

        print(f'Processing file in folder {root_folder}')
        #open the patch_names file in the folder and read it line by line to split the images
        with open('ca_patch_names.txt', 'r') as fp:
            while True:
                patch = fp.readline().strip()

                if not patch:
                    break
                month = get_month(patch)
                print(f'month: {month}')
                split_convert2TFrecord(tfwriter,
                    patch,
                    month,
                    os.path.join(root_folder, patch))
    except Exception as e:
        print(f"ERROR: TFRecord writer is not able to write files {str(e)}")

    tfwriter.close()

ROOT_FOLDER = '/home/cagastya/hdd/gee_central_valley'
OUT_FOLDER = '/home/cagastya/MIDS_Capstone/data/Tfrecords.CentralValley'

if __name__ == "__main__":
    create_split(ROOT_FOLDER, OUT_FOLDER)
