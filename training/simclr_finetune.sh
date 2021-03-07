#!/bin/bash

DATA_DIR=/data
# OUTPUT_DIR=/data/irrigation_data/models/simclr_finetune
OUTPUT_DIR=/mnt/irrigation_data/models/simclr_finetune
EPOCHS=50
BATCH_SIZE=32

S3_DIR="/mnt/irrigation_data/BigEarthNet_tfrecords_balanced"
DATA_DIR=/data/balanced

TRAIN_PERCENT=3
TRAIN_PERCENT_DIR="tfrecords_${TRAIN_PERCENT}_percent"
TRAIN_TARBALL="${S3_DIR}/${TRAIN_PERCENT_DIR}.tar"
TRAIN_DATA="${DATA_DIR}/${TRAIN_PERCENT_DIR}/train"
VAL_DATA="${DATA_DIR}/${TRAIN_PERCENT_DIR}/val"

TEST_TARBALL="${S3_DIR}/${TRAIN_PERCENT_DIR}.tar"
TEST_DATA="${DATA_DIR}/tfrecords_test/test"

GREEN='\033[0;32m'
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

function extract_training_data() {
  ### extract tfrecords
  mkdir -pm 777 "$DATA_DIR"
  log "extracting $TRAIN_TARBALL"
  sudo ls -la "$TRAIN_TARBALL"
  sudo tar xf "$TRAIN_TARBALL" \
    -C "$DATA_DIR" \
    --exclude="${TRAIN_PERCENT_DIR}/train.tfrecord" \
    --exclude="${TRAIN_PERCENT_DIR}/val.tfrecord" \
    --exclude="${TRAIN_PERCENT_DIR}/test*" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions
}

function extract_test_data() {
  ### extract tfrecords
  mkdir -pm 777 "$DATA_DIR"
  log "extracting $TEST_TARBALL"
  sudo ls -la "$TEST_TARBALL"
  sudo tar xf "$TEST_TARBALL" \
    -C "$DATA_DIR" \
    --exclude="tfrecords_test/test.tfrecord" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions
}

function remove_container() {
  log "Removing docker container: $DOCKER_NAME"
  docker rm -f "$DOCKER_NAME" 2>/dev/null || true
}

function simclr_finetune() {
  mkdir -pm 777 ${OUTPUT_DIR}
  trap remove_container EXIT
  for MODEL in $(ls -1 /mnt/irrigation_data/models/simclr_pretrain/*.h5); do
    local outfile="${TRAIN_PERCENT}_$(basename $MODEL .h5)"
    if [[ -f "${OUTPUT_DIR}/${outfile}.h5" ]]; then
      if [[ "${FORCE:-}" != "true" ]]; then
        warn "${outfile} already trained, skipping..."
        continue
      fi
      warn "${outfile} will be overwritten"
    fi
    log "Running training on $TRAIN_DATA"
    log "Pretrained model: $MODEL"
    model_dir=$(dirname "$MODEL")
    DOCKER_NAME="training_$(date '+%Y%m%d%H%M%S')"
    docker run --gpus all --name "$DOCKER_NAME" \
      --user $(id -u):$(id -g) \
      -v "$(pwd):/capstone_fall20_irrigation" \
      -v "$OUTPUT_DIR:$OUTPUT_DIR" \
      -v "$DATA_DIR:$DATA_DIR" \
      -v "$model_dir:$model_dir" \
      -w /capstone_fall20_irrigation \
      imander/irgapp \
      python3 finetune.py \
      --model "$MODEL" \
      -e "$EPOCHS" \
      -b "$BATCH_SIZE" \
      -o "${outfile}" \
      --output-dir "$OUTPUT_DIR" \
      --training-data "$TRAIN_DATA" \
      --validation-data "$VAL_DATA" \
      --test-data "$TEST_DATA"
    docker logs "$DOCKER_NAME" 2>&1 >"${OUTPUT_DIR}/${outfile}.log"
    remove_container
  done
}

[ -d "$TRAIN_DATA" ] || extract_training_data
[ -d "$TEST_DATA" ] || extract_test_data

FORCE=false
simclr_finetune
