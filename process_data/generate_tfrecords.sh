#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m' # No Color

EARTHNET_ROOT="/data/BigEarthNet-v1.0"
PROCESSING_DIR="/data/process_data"
SPLIT_DIR="balanced_splits_expanded"
S3_BUCKET="s3://mids-capstone-irrigation-detection/BigEarthNet_tfrecords_balanced_new"

function log() {
  msg=${1:-}
  printf "${GREEN}+${NC} ${msg}\n"
}

function sample_splits() {
  sample=${1:-}
  echo $sample
  for split in train test val; do
    cat splits/train.csv | awk -v sample=$sample 'BEGIN {srand()} !/^$/ { if (rand() <= sample/100) print $0}' >small_splits/${split}.csv
  done
}

function shard_records() {
  docker run --user "$(id -u):$(id -g)" --rm \
    -v "${PROCESSING_DIR}:/process_data" \
    -v $(pwd):/work -w /work imander/irgapp \
    python3 shard_data.py
}

function generate_tfrecords() {
  log "Initializing tfrecords directory"
  mkdir -p "${PROCESSING_DIR}/tfrecords"
  rm -rf "${PROCESSING_DIR}/tfrecords/*"

  docker run --user "$(id -u):$(id -g)" --rm \
    -v "${PROCESSING_DIR}:/process_data" \
    -v "${EARTHNET_ROOT}:/earthnet_data" \
    -v $(pwd):/work \
    -w /work \
    imander/irgapp \
    python3 prep_splits.py \
    -r /earthnet_data \
    -o /process_data/tfrecords \
    -n $@ \
    -l tensorflow

  echo
  log "Sharding tfrecords files"
  for file in $@; do
    data_type=$(basename "$file")
    mkdir -p "${PROCESSING_DIR}/tfrecords/${data_type}"
  done
  shard_records

  log "Creating tarball: ${OUT_DIR}.tar"
  rm -rf "${PROCESSING_DIR}/$OUT_DIR"
  mv "${PROCESSING_DIR}/tfrecords" "${PROCESSING_DIR}/$OUT_DIR"
  tar -C "$PROCESSING_DIR" -cf "${PROCESSING_DIR}/${OUT_DIR}.tar" "$OUT_DIR"

  log "Uploading to S3: ${S3_BUCKET}/${OUT_DIR}.tar"
  aws s3 cp "${PROCESSING_DIR}/$OUT_DIR.tar" "${S3_BUCKET}/"

  log "Cleaning Up"
  rm -f "${PROCESSING_DIR}/$OUT_DIR.tar" tfrecords
}

mkdir -p "$PROCESSING_DIR"
rm -rf "temp_splits"
for fraction in 1 3 10 25 50 100; do
  mkdir -p "temp_splits"
  cp "${SPLIT_DIR}/train_$fraction" temp_splits/train
  cp "${SPLIT_DIR}/val_$fraction" temp_splits/val
  OUT_DIR="tfrecords_${fraction}_percent"
  TRAIN="temp_splits/train"
  VAL="temp_splits/val"

  log "Generating sample splits on $fraction percent of data"
  generate_tfrecords $TRAIN $VAL
  rm -rf "temp_splits"
done

log "Processing test data set"
OUT_DIR="tfrecords_test"
generate_tfrecords "${SPLIT_DIR}/test"

exit 0
