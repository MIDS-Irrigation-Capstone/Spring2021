#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

GREEN='\033[0;32m'
NC='\033[0m' # No Color
OUT_DIR="$(pwd)/balanced_splits"

function log() {
  msg=${1:-}
  printf "${GREEN}+${NC} ${msg}" | cut -d. -f1
}

function calc() { awk "BEGIN { print $* }" | cut -f1 -d.; }

mkdir -p "$OUT_DIR"
mkdir -p tmp
pushd tmp

log "Splitting irrigated and non-irrigated records"
grep "Permanently irrigated land" ../big_earth_net_labels.txt | awk '{print $ 1}' | shuf >irrigated_records
RECORD_LEN=$(wc -l <irrigated_records)
grep -vf irrigated_records ../big_earth_net_labels.txt | awk '{print $ 1}' | shuf -n "$RECORD_LEN" >non_irrigated_records

log "Calculating splits"
TRAIN_LEN=$(calc "${RECORD_LEN}*0.7")
TEST_LEN=$(calc "(${RECORD_LEN}-${TRAIN_LEN})/2")
VAL_LEN=$TEST_LEN

log "Creating $(calc ${TRAIN_LEN}*2) training records"
shuf -n "$TRAIN_LEN" irrigated_records >irr_train
shuf -n "$TRAIN_LEN" non_irrigated_records >non_irr_train
cat irr_train non_irr_train | shuf >"${OUT_DIR}/train_100"

log "Creating $(calc ${TEST_LEN}*2) test and validation records"
grep -vf irr_train irrigated_records >irr_test_val
grep -vf non_irr_train non_irrigated_records >non_irr_test_val
shuf -n "$TEST_LEN" irr_test_val >irr_test
shuf -n "$TEST_LEN" non_irr_test_val >non_irr_test
cat irr_test non_irr_test | shuf >"${OUT_DIR}/test"

grep -vf irr_test irr_test_val >irr_val
grep -vf non_irr_test non_irr_test_val >non_irr_val
cat irr_val non_irr_val | shuf >"${OUT_DIR}/val_100"

for split in 1 3 10 25 50; do
  log "Creating split of ${split} percent records"
  len=$(calc "${TRAIN_LEN}*${split}/100")
  shuf -n "$len" irr_train >s1
  shuf -n "$len" non_irr_train >s2
  cat s1 s2 | shuf >"${OUT_DIR}/train_${split}"

  len=$(calc "${VAL_LEN}*${split}/100")
  shuf -n "$len" irr_val >s1
  shuf -n "$len" non_irr_val >s2
  cat s1 s2 | shuf >"${OUT_DIR}/val_${split}"
done

popd
# rm -rf tmp

exit 0
