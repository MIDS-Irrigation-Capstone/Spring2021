# This is a test script to check what has been pickled for the image patch files
ROOT='/home/cagastya/hdd/CroplandDataSet/Sentinel2/Irrigated/S2SR_0_2019_35.22734228_-111.5437402'
FILENAME = 'S2SR_0_2019_35.22734228_-111.5437402.pkl'
RP1 = '/hdd/BigEarthNet-v1.0/S2A_MSIL2A_20170717T113321_28_87'
FN1 = 'S2A_MSIL2A_20170717T113321_28_87_B01.tif'

import pickle
from os import path
import rasterio
import numpy as np

def read_pickle_file():
    with open (path.join(ROOT, FILENAME), 'rb') as fp:
        data = pickle.load(fp)

        print(f'Dictionary has rows: {len(data)}')

        for idx, key in enumerate(data):
            print(f'{data[key]}')
            if idx == 0:
                break

def read_from_image():
    band_ds = rasterio.open(path.join(RP1, FN1))
    band_data = np.array(band_ds.read(1))
    print(band_data)

if __name__ == "__main__":
    print('\n---From GEE---------------------')
    read_pickle_file()
    print('\n---From BEN---------------------')
    read_from_image()
