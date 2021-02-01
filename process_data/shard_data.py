#!/usr/bin/env python
import tensorflow as tf

for dataset in ["train", "test", "val"]:
    raw_dataset = tf.data.TFRecordDataset(f"./tfrecords/{dataset}.tfrecord")

    shards = 50

    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(
            f"./tfrecords/{dataset}/part-{i}.tfrecord"
        )
        writer.write(raw_dataset.shard(shards, i))
