# Preparing BigEarthNet Data

Processing scripts were taken from the [BigEarthNet](https://gitlab.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/tree/master) repository.

## Prerequisites

- Docker installed
- BigEarthNet data downloaded and extracted
- Sufficient disk space to create tar archives (`100GB+`)
- AWS cli installed and configured with permissions to write to S3

## Configuration

The following variables can be modified in [generate_tfrecords.sh](./generate_tfrecords.sh) to suit your environment.

```
EARTHNET_ROOT="Path to extracted BigEarthNet data"
S3_BUCKET="S3 bucket path with key where tfrecords will be written to"
```

## Convert data

Run `./generate_tfrecords.sh` to begin process the BigEarthNet data and creating `tfrecords` files. By default subsets of 1, 3, 10, and 25 percent of the full dataset will be converted to `tfrecords` files.
