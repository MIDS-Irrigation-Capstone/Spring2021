#!/usr/bin/env python
import os
import tensorflow as tf

for dataset in ["train", "val", "test"]:
    ds_file = f"/process_data/tfrecords/{dataset}.tfrecord"
    if not os.path.isfile(ds_file):
        continue
    raw_dataset = tf.data.TFRecordDataset(ds_file)

    shards = 50

    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(
            f"/process_data/tfrecords/{dataset}/part-{i}.tfrecord"
        )
        writer.write(raw_dataset.shard(shards, i))
