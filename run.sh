#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

function proceed() {
  echo
  read -p "Proceed? [Y/n]: " -n 1 -r
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
  ARCH=$(selectWithDefault 'ResNet50' 'InceptionV3' 'Xception', 'ResNet101V2')
}

function get_epochs() {
  read -p "Number of epochs [50]: " EPOCHS
  EPOCHS=${EPOCHS:-50}
}

function get_batch_size() {
  read -p "Batch size [32]: " BATCH_SIZE
  BATCH_SIZE=${BATCH_SIZE:-32}
}

function get_augment() {
  AUGMENT=True
  read -p "Augment images [yes]: " -n 1 -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    AUGMENT=False
  fi
}

function get_ouput_prefix() {
  local default=${1:-default}
  read -p "Output file prefix: " OUTPUT_PREFIX
  OUTPUT_PREFIX=${OUTPUT_PREFIX:-$default}
}

function baseline_training() {
  get_architecture
  get_epochs
  get_batch_size
  get_augment
  get_ouput_prefix
  cat <<EOF

Architecture:   $ARCH
Epochs:         $EPOCHS
Batch Size:     $BATCH_SIZE
Augment Images: $AUGMENT
Outfile Prefix: $OUTPUT_PREFIX
EOF
  proceed
  docker run --rm -v "$(pwd):/capstone_fall20_irrigation" -v /mnt/irrigation_data:/data imander/irgapp \
    python3 supervised_classification.py -a $ARCH -o $OUTPUT_PREFIX -e $EPOCHS -b $BATCH_SIZE -g $AUGMENT
}

baseline_training

exit 0
