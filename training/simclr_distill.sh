#!/bin/bash


OUTPUT_DIR=/mnt/irrigation_data/models/CA_distill
EPOCHS=50
BATCH_SIZE=32

MODEL_DIR="/mnt/irrigation_data/models/CA_finetune"

SET=balanced
#SET=expanded # Don't forget to add --expanded-labels!
DATA_DIR="/data/${SET}"

TEST_DATA="${DATA_DIR}/tfrecords_test/test"

TRAIN_ARCHS="ResNet50 InceptionV3 Xception ResNet101V2 ResNet152"


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


function remove_container() {
  log "Removing docker container: $DOCKER_NAME"
  docker rm -f "$DOCKER_NAME" 2>/dev/null || true
}

function simclr_distill() {
  mkdir -pm 777 ${OUTPUT_DIR}
  trap remove_container EXIT
  for MODEL in $(ls -1 /mnt/irrigation_data/models/CA_finetune/${SET}*.h5); do

    for ARCH in $TRAIN_ARCHS; do
      local outfile="$(basename $MODEL .h5)_${ARCH}"
      if [[ -f "${OUTPUT_DIR}/${outfile}.h5" ]]; then
        if [[ "${FORCE:-}" != "true" ]]; then
          warn "${outfile} already trained, skipping..."
          continue
        fi
        warn "${outfile} will be overwritten"
      fi
      log "Running training on $TEST_DATA"
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
        python3 distill_trainer.py \
        --model "$MODEL" \
        -e "$EPOCHS" \
        -b "$BATCH_SIZE" \
        -o "${outfile}" \
        --output-dir "$OUTPUT_DIR" \
        --train_data "$TEST_DATA" \
        --arch "$ARCH" 
      docker logs "$DOCKER_NAME" 2>&1 >"${OUTPUT_DIR}/${outfile}.log"
      remove_container
    done
  done
}


FORCE=false
simclr_distill
