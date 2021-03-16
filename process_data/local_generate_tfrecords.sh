#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m' # No Color

EARTHNET_ROOT="/hdd/BigEarthNet-v1.0"
OUT_PATH="/home/cagastya/MIDS_Capstone/data/Tfrecords.SimCLR/"
#S3_BUCKET="s3://mids-capstone-irrigation-detection/BigEarthNet_tfrecords"

function log() {
  msg=${1:-}
  printf "${GREEN}+${NC} ${msg}\n"
}

# function sample_splits() {
#   sample=${1:-}
#   echo $sample
#   for split in train test val; do
#     cat splits/train.csv | awk -v sample=$sample 'BEGIN {srand()} !/^$/ { if (rand() <= sample/100) print $0}' >small_splits/${split}.csv
#   done
# }

# function generate_tfrecords() {
#   docker run --user "$(id -u):$(id -g)" --rm -v "${EARTHNET_ROOT}:/earthnet_data" -v $(pwd):/work -w /work imander/irgapp \
#     python3 prep_splits.py \
#     -r /earthnet_data \
#     -o tfrecords \
#     -n "small_splits/train.csv" "small_splits/test.csv" "small_splits/val.csv" \
#     -l tensorflow
# }

function generate_tfrecords_locally() {
  #docker run --user "$(id -u):$(id -g)" --rm -v "${EARTHNET_ROOT}:/earthnet_data" -v $(pwd):/work -w /work imander/irgapp \
    python3 prep_splits.py \
    -r ${EARTHNET_ROOT} \
    -o ${OUT_PATH} \
    -n "pretraining_splits/train"\
    -l tensorflow
}

# function shard_records() {
#   docker run --user "$(id -u):$(id -g)" --rm -v $(pwd):/work -w /work imander/irgapp \
#     python3 shard_data.py
# }

#mkdir -p /hdd/tfrecords

# function create_samples() {
#   for fraction in 1 3 10 25; do
#     #make sure we sample them as stratified splits
#     log "Generating sample splits on $fraction percent of data"
#     sample_splits $fraction
#
#     log "Initializing tfrecords directory"
#     mkdir -p tfrecords
#     rm -rf tfrecords/*
#
#     log "Generating sample splits on $fraction percent of data"
#     generate_tfrecords
#
#     #log "Sharding tfrecords files"
#     #mkdir -p tfrecords/{train,test,val}
#     #shard_records
#
#     #log "Creating tarball: ${records}.tar"
#     #records="tfrecords_${fraction}_percent"
#     #mv tfrecords "$records"
#     #tar -cf "${records}.tar" "$records"
#
#     #log "Moving to destination folder: ${S3_BUCKET}/${records}.tar"
#     #aws s3 cp "$records.tar" "${S3_BUCKET}/"
#
#     #log "Cleaning Up"
#     #rm -f "$records.tar" tfrecords
#   done
# }

function convert_all() {
  log "Converting data to tfrecords"
  generate_tfrecords_locally
}

convert_all

exit 0
