#!/usr/bin/env python
import os
import tensorflow as tf

ROOT='/home/cagastya/MIDS_Capstone/data/Tfrecords.CentralValley'
ds_file = f"{ROOT}/CV.tfrecord"


if not os.path.isfile(ds_file):
   print('Cannot find the file CV.tfrecord')
else:
    raw_dataset = tf.data.TFRecordDataset(ds_file)

    shards = 50

    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(
            f"{ROOT}/cv_data_parts/part-{i}.tfrecord"
        )
        writer.write(raw_dataset.shard(shards, i))
