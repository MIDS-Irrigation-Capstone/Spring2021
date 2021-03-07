#!/bin/bash

DATA_DIR=/data
OUTPUT_DIR=/data/irrigation_data/models
ARCH=ResNet152
OUTPUT_PREFIX="1_percent_${ARCH}_E50_B32"
OUTPUT_DIR=/mnt/irrigation_data/models/simclr_pretrain
EPOCHS=50
BATCH_SIZE=32

TRAIN_DIR=/data/unbalanced/tfrecords
TRAIN_PERCENT=1
TRAIN_PERCENT_DIR="tfrecords_${TRAIN_PERCENT}_percent"
TRAIN_TARBALL="/mnt/irrigation_data/BigEarthNet_tfrecords/${TRAIN_PERCENT_DIR}.tar"
TRAIN_DATA="${TRAIN_DIR}/${TRAIN_PERCENT_DIR}/train"

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

function extract_training_data() {
  ### extract tfrecords
  mkdir -pm 777 "$TRAIN_DIR"
  log "extracting $TRAIN_TARBALL"
  sudo ls -la "$TRAIN_TARBALL"
  sudo tar xf "$TRAIN_TARBALL" \
    -C "$TRAIN_DIR" \
    --exclude="${TRAIN_PERCENT_DIR}/train.tfrecord" \
    --exclude="${TRAIN_PERCENT_DIR}/val*" \
    --exclude="${TRAIN_PERCENT_DIR}/test*" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions
}

function remove_container() {
  log "Removing docker container: $DOCKER_NAME"
  docker rm -f $DOCKER_NAME 2>/dev/null || true
}

function simclr_pretrain() {
  mkdir -pm 777 ${OUTPUT_DIR}
  trap remove_container EXIT
  for AUG in $AUGMENTATIONS; do
    local outfile="${OUTPUT_PREFIX}_${AUG}"
    if [[ -f "${OUTPUT_DIR}/${outfile}.log" ]]; then
      if [[ "${FORCE:-}" != "true" ]]; then
        warn "${outfile} already trained, skipping..."
        continue
      fi
      warn "${outfile} will be overwritten"
    fi
    log "Running training on $TRAIN_DATA"
    log "Augmentations: $AUG"
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
      --augmentations "$AUG" \
      --train-data "$TRAIN_DATA"
    docker logs "$DOCKER_NAME" >"${OUTPUT_DIR}/${outfile}.log"
    remove_container
  done
}

[ -d "$TRAIN_DATA" ] || extract_training_data

read -r -d '' IAN <<'EOF'
rotation,shift
rotation,flip
rotation,zoom
rotation,blur
rotation,brightness
rotation,contrast
rotation,gain
shift,flip
shift,zoom
shift,blur
EOF

read -r -d '' CHITRA <<'EOF'
shift,brightness
shift,contrast
shift,gain
flip,zoom
flip,blur
flip,brightness
flip,contrast
flip,gain
zoom,blur
EOF

read -r -d '' SIRAK <<'EOF'
zoom,brightness
zoom,contrast
zoom,gain
blur,brightness
blur,contrast
blur,gain
brightness,contrast
brightness,gain
contrast,gain
EOF

read -r -d '' SPECKLE <<'EOF'
speckle,shift
speckle,flip
speckle,zoom
speckle,blur
speckle,brightness
speckle,contrast
speckle,gain
speckle,rotation
EOF

FORCE=false
AUGMENTATIONS=$SPECKLE
simclr_pretrain
