import os
import csv
import json
import numpy as np
import tensorflow as tf

#SPLIT_NAMES = ['train.csv', 'test.csv', 'val.csv']
SPLIT_NAMES = ['train.csv', 'test.csv']
SPLIT_DIR = 'splits'
IMG_DIR = "/hdd/BigEarthNet-v1.0"
OUT_DIR = 'splits'

label_counts = {}
def get_patch_names():
    try:
        patch_names_list = []
        split_names = []
        for csv_file in SPLIT_NAMES:
            patch_names_list.append([]) # a list for each split
            split_names.append(os.path.basename(csv_file).split('.')[0])
            with open(f'{SPLIT_DIR}/{csv_file}', 'r') as fp:
                csv_reader = csv.reader(fp, delimiter=',')
                for row in csv_reader:
                    patch_names_list[-1].append(row[0].strip())
        return split_names, patch_names_list
    except:
        print('ERROR: some csv files either do not exist or have been corrupted')
        exit()


def create_split_labels(patch_names, label_indices, fp):
    progress_bar = tf.keras.utils.Progbar(target=len(patch_names))
    for patch_idx, patch_name in enumerate(patch_names):
        patch_folder_path = os.path.join(IMG_DIR, patch_name)

        patch_json_path = os.path.join(
            patch_folder_path, patch_name + '_labels_metadata.json')

        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)

        original_labels = patch_json['labels']
        fp.write(f'{patch_name}, {original_labels}\n')
        for label in original_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # update the progress bar
        progress_bar.update(patch_idx)
    return label_counts

with open('label_indices.json', 'rb') as f:
    label_indices = json.load(f)

if __name__ == "__main__":
    split_names, patch_names_list = get_patch_names()

    # for each split in the dir, do below
    for split_idx in range(len(patch_names_list)):
        with open(f'{OUT_DIR}/{split_names[split_idx]}_labels.csv', 'w') as fp:
            label_counts = create_split_labels(patch_names_list[split_idx], label_indices, fp)
            if label_counts:
                with open(f'{OUT_DIR}/{split_names[split_idx]}_counts.json', 'w') as cfp:
                    json.dump(label_counts, cfp)
