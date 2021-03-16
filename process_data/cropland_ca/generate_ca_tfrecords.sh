#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m' # No Color

#CA_ROOT="/mnt/irrigation_data/gee_california/"
CA_ROOT="/mnt/irrigation_data/gee_central_valley/"
PROCESSING_DIR="/home/ubuntu/repos/Spring2021/process_data/cropland_ca/"
S3_BUCKET="s3://mids-capstone-irrigation-detection/CA_tfrecords"

function log() {
  msg=${1:-}
  printf "${GREEN}+${NC} ${msg}\n"
}

function shard_records() {
  docker run --user "$(id -u):$(id -g)" --rm \
    -v "${PROCESSING_DIR}:/process_data/cropland_ca" \
    -v $(pwd):/work -w /work imander/irgapp \
    python3 shard_data.py
}

function generate_tfrecords() {
  log "Initializing tfrecords directory"
  mkdir -p "${PROCESSING_DIR}/tfrecords"
  rm -rf "${PROCESSING_DIR}/tfrecords/*"

  docker run --user "$(id -u):$(id -g)" --rm \
    -v "${PROCESSING_DIR}:/process_data" \
    -v "${CA_ROOT}:/ca_data" \
    -v $(pwd):/work \
    -w /work \
    imander/irgapp \
    python3 split_ca_image.py

  echo
  log "Sharding tfrecords files"
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

log "Processing CA data set"
OUT_DIR="tfrecords_ca"
generate_tfrecords

exit 0
