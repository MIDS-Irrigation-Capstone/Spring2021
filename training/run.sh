#!/bin/bash

set -euo pipefail
IFS=$'\n\t '

DATA_DIR=/data
DATA_TYPE='expanded'
EXPANDED_LABELS="--expanded-labels"

S3_DIR=/mnt/irrigation_data
# OUTPUT_DIR=/data/irrigation_data/models
OUTPUT_DIR=/mnt/irrigation_data/models

# Defaults
EPOCHS=50
BATCH_SIZE=32
AUGMENT=False
AUGMENT_STR=noaug
WEIGHTS=False

DATE=$(date '+%Y%m%d')
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

function proceed() {
  local pretrain=False
  local data_set="Expanded Labels"

  if [[ -n "${PRETRAIN:-}" ]]; then
    pretrain=True
  fi
  if [[ "${DATA_TYPE}" == "balanced" ]]; then
    data_set="Exact Labels"
  fi
  cat <<EOF

  Architecture:    ${ARCH:-all}
  Data Set:        $data_set
  Split Percent:   ${SPLIT_PERCENT:-all}
  Epochs:          $EPOCHS
  Batch Size:      $BATCH_SIZE
  Augment Images:  $AUGMENT
  Apply Weights:   $WEIGHTS
  Use Pretrain:    $pretrain
  Output path:     $OUTPUT_DIR
  Added Labels:    ${LABELS:-None}
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
  ARCH=$(selectWithDefault 'all' 'ResNet50' 'ResNet152' 'InceptionV3' 'Xception' 'ResNet101V2')
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
  read -e -p "Model output directory [$OUTPUT_DIR]: " out
  OUTPUT_DIR=${out:-$OUTPUT_DIR}
  read -e -p "Additional Labels: " LABELS
}

function get_data_split() {
  echo "BigEarthNet data split percentage"
  SPLIT_PERCENT=$(selectWithDefault 'all' '1 percent' '3 percent' '10 percent' '25 percent' '50 percent' '100 percent' | awk '{print $1}')
}

function get_weights() {
  read -e -p "Apply weights [no]: " -r
  if [[ $REPLY =~ ^[Yy].*$ ]]; then
    WEIGHTS='True'
  fi
}

function get_dataset() {
  read -e -p "Expanded data set [yes]: " -r
  if [[ $REPLY =~ ^[Nn].*$ ]]; then
    DATA_TYPE='balanced'
    DATA_SET='Exact Labels'
    EXPANDED_LABELS=""
  fi
}

function fetch_data() {
  local out_dir=$(dirname "$TF_RECORDS")
  mkdir -p "$out_dir"

  local records="BigEarthNet_tfrecords_balanced"
  if [[ "$DATA_TYPE" == "expanded" ]]; then
    records="${records}_expanded"
  fi

  log "Extracting training data"
  sudo tar xf "${S3_DIR}/${records}/tfrecords_${SPLIT_PERCENT}.tar" \
    -C "$out_dir" \
    --exclude="tfrecords_${SPLIT_PERCENT}/train.tfrecord" \
    --exclude="tfrecords_${SPLIT_PERCENT}/val.tfrecord" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions

  [[ -d "${out_dir}/tfrecords_test/test" ]] && return

  log "Extracting test data"
  sudo tar xf "${S3_DIR}/${records}/tfrecords_test.tar" \
    -C "$out_dir" \
    --exclude="tfrecords_${SPLIT_PERCENT}/test.tfrecord" \
    --owner=ubuntu \
    --group=ubuntu \
    --no-same-permissions
}

function set_vars() {
  TF_RECORDS="${DATA_DIR}/${DATA_TYPE}/tfrecords"
  OUTPUT_PREFIX="${SPLIT_PERCENT}_${ARCH}_E${EPOCHS}_B${BATCH_SIZE}_${AUGMENT_STR}"
  if [[ -n "${LABELS:-}" ]]; then
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_${LABELS}"
  fi
  OUTPUT_PREFIX="${OUTPUT_PREFIX}-${DATE}"
  TRAIN_DATA="${TF_RECORDS}/train"
  VAL_DATA="${TF_RECORDS}/val"

  MODEL_DIR="$MODEL_TYPE"
  if [[ "$DATA_TYPE" == "expanded" ]]; then
    MODEL_DIR="${MODEL_TYPE}_ex"
  fi
}

function prepare() {
  mkdir -p "${OUTPUT_DIR}/${MODEL_TYPE}"
  if [[ ! -d "${TF_RECORDS}_${SPLIT_PERCENT}" ]]; then
    warn "${TF_RECORDS}_${SPLIT_PERCENT} not found"
    fetch_data
  fi

  log "Creating symbolic link"
  rm -f "${TF_RECORDS}"
  ln -s "${TF_RECORDS}_${SPLIT_PERCENT}" "${TF_RECORDS}"
}

function prompt_settings() {
  get_architecture
  get_epochs
  get_batch_size
  get_augment
  get_data_split
  get_weights
  get_ouput_info
  get_dataset
}

function remove_container() {
  log "Removing docker container: $DOCKER_NAME"
  docker rm -f $DOCKER_NAME || true
}

function baseline_training() {
  TEST_DIR="${DATA_DIR}/${DATA_TYPE}/tfrecords_test/test"
  DOCKER_NAME="training_$(date '+%Y%m%d%H%M%S')"

  trap remove_container EXIT
  docker run --gpus all --name "$DOCKER_NAME" \
    --user $(id -u):$(id -g) \
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
    --validation-set "$VAL_DATA" \
    ${EXPANDED_LABELS} \
    ${EXTRA_ARGS:-}
  docker logs "$DOCKER_NAME" >"${OUTPUT_DIR}/${MODEL_DIR}/${OUTPUT_PREFIX}.log"
  remove_container
}

function train() {
  TRAIN_DATA="${DATA_DIR}/train"
  VAL_DATA="${DATA_DIR}/val"
  local arch=${ARCH:-}
  if [[ -z "$arch" ]] || [[ "$arch" == "all" ]]; then
    arch="ResNet50 InceptionV3 Xception ResNet101V2"
  fi
  split=${SPLIT_PERCENT:-}
  if [[ -z "$split" ]] || [[ "$split" == "all" ]]; then
    split="1 3 10 25 50 100"
  fi
  for SPLIT in $(echo $split); do
    SPLIT_PERCENT="${SPLIT}_percent"
    for ARCH in $(echo $arch); do
      set_vars
      if [[ -f "${OUTPUT_DIR}/${MODEL_DIR}/${OUTPUT_PREFIX}.log" ]]; then
        if [[ "${FORCE:-}" != "true" ]]; then
          warn "${OUTPUT_PREFIX} already trained, skipping..."
          continue
        fi
      fi
      log "Training: ${OUTPUT_PREFIX}"
      prepare
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
  '-f' | '--force')
    FORCE=true
    shift
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
  MODEL_TYPE=supervised_baseline
  TRAIN_FUNCTION=baseline_training
  ;;
'baseline-pretrained')
  MODEL_TYPE=supervised_baseline_pretrained
  TRAIN_FUNCTION=baseline_training
  PRETRAIN=True
  EXTRA_ARGS="--pretrained"
  ;;
'NONE')
  echo "Must specify model with -m or --model option"
  exit 1
  ;;
*)
  cat <<EOF
Invalid model option!
Choose one of:
   - baseline
   - baseline-pretrained
EOF
  exit 1
  ;;
esac

if [[ -z "${PROMPT:-}" ]]; then
  prompt_settings
fi
proceed
train

exit 1
