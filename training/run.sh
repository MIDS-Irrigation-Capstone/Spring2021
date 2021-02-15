#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

DATA_DIR=/data/tfrecords
TEST_DIR=/data/tfrecords_test/test
S3_DIR=/mnt/irrigation_data/BigEarthNet_tfrecords
# OUTPUT_DIR=/data/irrigation_data/models
OUTPUT_DIR=/mnt/irrigation_data/models

# Defaults
EPOCHS=50
BATCH_SIZE=32
AUGMENT=False
AUGMENT_STR=noaug
WEIGHTS=False

GREEN='\033[0;32m'
NC='\033[0m' # No Color

function log() {
  msg=${1:-}
  printf "$(date -u) ${GREEN}INFO${NC} $msg\n"
}

function proceed() {
  cat <<EOF

  Architecture:    ${ARCH:-"all"}
  Epochs:          $EPOCHS
  Batch Size:      $BATCH_SIZE
  Augment Images:  $AUGMENT
  Apply Weights:   $WEIGHTS
  Output path:     $OUTPUT_DIR
EOF

  echo
  read -p "Proceed with training? [Y/n]: " -n 1 -r
  if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit
  fi
  echo
}

function selectWithDefault() {

  PS3="Choice [${@:1:1}]: "
  local item i=0 numItems=$#

  # Print numbered menu items, based on the arguments passed.
  for item; do # Short for: for item in "$@"; do
    printf '%s\n' "$((++i))) $item"
  done >&2 # Print to stderr, as `select` does.

  # Prompt the user for the index of the desired item.
  while :; do
    printf %s "${PS3-#? }" >&2 # Print the prompt string to stderr, as `select` does.
    read -r index
    # Make sure that the input is either empty or that a valid index was entered.
    [[ -z $index ]] && break # empty input
    ((index >= 1 && index <= numItems)) 2>/dev/null || {
      echo "Invalid selection. Please try again." >&2
      continue
    }
    break
  done

  # Output the selected item, if any.
  [[ -n $index ]] && printf %s "${@:index:1}" || echo "${@:1:1}"

}

function get_architecture() {
  echo "Choose Training Architecture"
  ARCH=$(selectWithDefault 'ResNet50' 'InceptionV3' 'Xception' 'ResNet101V2')
}

function get_epochs() {
  read -e -p "Number of epochs [50]: " EPOCHS
  EPOCHS=${EPOCHS:-50}
}

function get_batch_size() {
  read -e -p "Batch size [32]: " BATCH_SIZE
  BATCH_SIZE=${BATCH_SIZE:-32}
}

function get_augment() {
  AUGMENT=False
  AUGMENT_STR=noaug
  read -e -p "Augment images [no]: " -r
  if [[ $REPLY =~ ^[Yy].*$ ]]; then
    AUGMENT_STR=aug
    AUGMENT=True
  fi
}

function get_ouput_info() {
  local default=/mnt/irrigation_data/models
  read -e -p "Model output directory [$default]: " OUTPUT_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$default}

  default="${ARCH}_E${EPOCHS}_B${BATCH_SIZE}_${AUGMENT_STR}-$(date '+%Y%m%d')"
  read -e -p "Output file prefix: " OUTPUT_PREFIX
  if [[ -z "$OUTPUT_PREFIX" ]]; then
    OUTPUT_PREFIX=$default
  else
    OUTPUT_PREFIX="${OUTPUT_PREFIX}-${default}"
  fi
}

function get_training_dataset() {
  local default=/data/tfrecords/train.tfrecord
  read -e -p "Training dataset [$default]: " TRAIN_DATA
  TRAIN_DATA=${TRAIN_DATA:-$default}
}

function get_validation_dataset() {
  local default=/data/tfrecords/val.tfrecord
  read -e -p "Validation dataset [$default]: " VAL_DATA
  VAL_DATA=${TRAIN_DATA:-$default}
}

function get_data_split() {
  echo "BigEarthNet data split percentage"
  SPLIT_PERCENT=$(selectWithDefault '1 percent' '3 percent' '10 percent' '25 percent' '50 percent' '100 percent' | tr ' ' '_')
  OUTPUT_PREFIX="${SPLIT_PERCENT}_${ARCH}_E${EPOCHS}_B${BATCH_SIZE}_${AUGMENT_STR}-$(date '+%Y%m%d')"
  TRAIN_DATA="${DATA_DIR}/train"
  VAL_DATA="${DATA_DIR}/val"
}

function get_weights() {
  read -e -p "Apply weights [no]: " -r
  if [[ $REPLY =~ ^[Yy].*$ ]]; then
    WEIGHTS='True'
  fi
}

function prepare() {
  mkdir -p "$OUTPUT_DIR/${MODEL_DIR}"
  if [[ ! -d "${DATA_DIR}_${SPLIT_PERCENT}" ]]; then
    log "${DATA_DIR}_${SPLIT_PERCENT} not found"
    if [[ ! -f "${DATA_DIR}_${SPLIT_PERCENT}.tar" ]]; then
      log "Downloading training data from S3"
      sudo cp "${S3_DIR}/tfrecords_${SPLIT_PERCENT}.tar" /data/
      sudo chown ubuntu: "${DATA_DIR}_${SPLIT_PERCENT}.tar"
      chmod 644 "${DATA_DIR}_${SPLIT_PERCENT}.tar"
    fi
    log "Extracting training data"
    tar xf "${DATA_DIR}_${SPLIT_PERCENT}.tar" -C "$(dirname "$DATA_DIR")"
  fi
  log "Creating symbolic link"
  rm -f ${DATA_DIR}
  ln -s "${DATA_DIR}_${SPLIT_PERCENT}" ${DATA_DIR}
}

function prompt_settings() {
  get_architecture
  get_epochs
  get_batch_size
  get_augment
  get_data_split
  get_weights
  # get_ouput_info
  # get_training_dataset
  # get_validation_dataset
}

function baseline_training() {
  DOCKER_NAME="training_$(date '+%Y%m%d%H%M%S')"
  trap "docker rm $DOCKER_NAME" EXIT
  docker run --gpus all --name "$DOCKER_NAME" \
    -v "$(pwd):/capstone_fall20_irrigation" \
    -v "$OUTPUT_DIR:$OUTPUT_DIR" \
    -v "$DATA_DIR:$DATA_DIR" \
    -v "$TEST_DIR:/data/test" \
    -w /capstone_fall20_irrigation \
    imander/irgapp \
    python3 supervised_classification.py \
    -a "$ARCH" \
    -o "$OUTPUT_PREFIX" \
    --output-dir "$OUTPUT_DIR/$MODEL_DIR" \
    -e "$EPOCHS" \
    -b "$BATCH_SIZE" \
    -g "$AUGMENT" \
    --weights "$WEIGHTS" \
    --train-set "$TRAIN_DATA" \
    --validation-set "$VAL_DATA"
  docker logs "$DOCKER_NAME" >"${OUTPUT_DIR}/${MODEL_DIR}/${OUTPUT_PREFIX}.log"
}

function default_training() {
  TRAIN_DATA="${DATA_DIR}/train"
  VAL_DATA="${DATA_DIR}/val"
  for ARCH in ResNet50 InceptionV3 Xception ResNet101V2; do
    for split in 1 3 10 25 50 100; do
      SPLIT_PERCENT="${split}_percent"
      prepare
      OUTPUT_PREFIX="${SPLIT_PERCENT}_${ARCH}_E${EPOCHS}_B${BATCH_SIZE}_${AUGMENT_STR}-$(date '+%Y%m%d')"
      $TRAIN_FUNCTION
    done
  done
}

TEMP=$(getopt -o 'm:' --long 'no-prompt,model:' -- "$@")

if [ $? -ne 0 ]; then
  echo 'Terminating...' >&2
  exit 1
fi

# Note the quotes around "$TEMP": they are essential!
eval set -- "$TEMP"
unset TEMP

while true; do
  case "$1" in
  '--no-prompt')
    PROMPT=false
    shift
    continue
    ;;
  '-m' | '--model')
    MODEL="$2"
    shift 2
    continue
    ;;
  '--')
    shift
    break
    ;;
  esac
done

case "${MODEL:-NONE}" in
'baseline')
  MODEL_DIR=supervised_baseline
  TRAIN_FUNCTION=baseline_training
  ;;
'NONE')
  echo "Must specify model with -m or --model option"
  exit 1
  ;;
*)
  echo -e "Invalid model option!\nChoose one of (baseline)"
  exit 1
  ;;
esac

if [[ -z "${PROMPT:-}" ]]; then
  prompt_settings
  proceed
  prepare
  $TRAIN_FUNCTION
else
  proceed
  default_training
fi

exit 0
