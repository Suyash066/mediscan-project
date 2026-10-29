#!/usr/bin/env bash
# Reproduces every figure and table.   bash run_all.sh
#   MEDISCAN_DATA=/path/to/data   (default ./data; see README for layout)
#   QUICK=1   tiny smoke-test settings (2 epochs, 64px, 3 tuning trials) - numbers are meaningless, it only proves the pipeline runs
set -euo pipefail
cd "$(dirname "$0")"
export MEDISCAN_DATA="${MEDISCAN_DATA:-data}"
PY="${PYTHON:-python}"
mkdir -p results
$PY -m pip freeze > results/pip_freeze.txt 2>/dev/null || true
if [ "${QUICK:-0}" = "1" ]; then
  export MEDISCAN_NO_PRETRAINED=1
  TR="--epochs 2 --res 64"; TUNE="--trials 3 --search_epochs 2 --final_epochs 3 --res 64"; EV="--epochs 2 --res 64"
else
  TR=""; TUNE="--trials ${TRIALS:-12}"; EV=""
fi
echo "### PART 1"
$PY audit.py
$PY split.py
$PY baselines.py
$PY arch_diagram.py
$PY train.py --exps transfer_resnet18 arogya arogya_noskip arogya_nose arogya_meta arogya_res96 arogya_augnone arogya_cw $TR
$PY aggregate.py
echo "### PART 2"
$PY p2_tune.py $TUNE
$PY p2_threshold.py
$PY p2_eval.py $EV
$PY p2_ensemble.py
$PY p2_embed.py
if [ -f results/external_lock.json ]; then
  echo "external hospital already evaluated once (results/external_lock.json) - skipping, as required"
else
  $PY p2_external.py --confirm-once ${FIX:+--fix $FIX} $EV
fi
$PY export.py
$PY tools/ood_tests.py
$PY ceo_arithmetic.py
$PY make_report.py
echo "DONE. Now write the <!-- WRITE --> interpretation slots in reports/Report.md and export to PDF (pandoc)."
