#!/bin/bash

DATA_DIR=/data
ARCH=ResNet152
OUTPUT_PREFIX="BigEarthNet_SimCLR_pretrain_${ARCH}_E50_B32_V2"
OUTPUT_DIR=/mnt/irrigation_data/models/simclr2_pretrain
EPOCHS=50
BATCH_SIZE=32
SAVE_ITER=1

TRAIN_DATA="/data/Tfrecord.SimCLR/train.tfrecord"

GREEN='\034[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

function log() {
  msg=${1:-}
  printf "$(date -u) ${GREEN}INFO${NC} $msg\n"
}

function warn() {
  msg=${1:-}
  printf "$(date -u) ${YELLOW}WARNING${NC} $msg\n"
}

function remove_container() {
  log "Removing docker container: $DOCKER_NAME"
  docker rm -f $DOCKER_NAME 2>/dev/null || true
}

function simclr_pretrain() {
  mkdir -pm 777 ${OUTPUT_DIR}
  trap remove_container EXIT
  local outfile="${OUTPUT_PREFIX}"


  log "Running training on $TRAIN_DATA"
  DOCKER_NAME="training_$(date '+%Y%m%d%H%M%S')"
  docker run --gpus all --name "$DOCKER_NAME" \
    --user $(id -u):$(id -g) \
    -v "$(pwd):/capstone_fall20_irrigation" \
    -v "$OUTPUT_DIR:$OUTPUT_DIR" \
    -v "$DATA_DIR:$DATA_DIR" \
    -w /capstone_fall20_irrigation \
    imander/irgapp \
    python3 simclr.py \
    -a "$ARCH" \
    -o "${outfile}" \
    --output-dir "$OUTPUT_DIR" \
    -e "$EPOCHS" \
    -b "$BATCH_SIZE" \
    --train-data "$TRAIN_DATA" \
    --save-iterations "$SAVE_ITER"
  docker logs "$DOCKER_NAME" >"${OUTPUT_DIR}/${outfile}.log"
  remove_container
}


simclr_pretrain
