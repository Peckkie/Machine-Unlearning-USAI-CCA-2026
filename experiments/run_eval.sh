#!/usr/bin/env bash
# Evaluate one trained model on Lab test (1312) + Field test (807), append to $RESULTS_CSV
# usage: bash experiments/run_eval.sh <original|unlearned|C1|C2> <seed> <gpu>
#        bash experiments/run_eval.sh paper-unlearned 1 <gpu>   # โมเดล unlearned ใน paper -> บันทึกเป็น unlearned seed 1
#        bash experiments/run_eval.sh paper-original  1 <gpu>   # โมเดล original ใน paper (legacy) -> บันทึกเป็น original_paper
#   ใช้ 2 คำสั่งนี้เช็คก่อนว่า evaluate.py ได้ตัวเลขตรง Table 3/4
set -uo pipefail
source "$(dirname "$0")/_common.sh"

MODEL=${1:?usage: run_eval.sh <model> <seed> <gpu>}
SEED=${2:?seed}
GPU=${3:?gpu}

require_vars SAVE_DIR
require_paths LAB_TEST_CSV FIELD_TEST_CSV
case $MODEL in
    paper-original)  H5=$PAPER_ORIGINAL_H5;  LABEL=original_paper ;;   # legacy pipeline: ไม่นับเป็น seed ของ original
    paper-unlearned) H5=$PAPER_UNLEARNED_H5; LABEL=unlearned ;;        # train.py pipeline เดียวกัน: นับเป็น unlearned seed 1
    *) setup_model "$MODEL" "$SEED"; H5=$R2_DIR/$R2_NAME.h5; LABEL=$MODEL ;;
esac
require_paths H5

activate_env
check_env
cd "$REPO_DIR" || die "ไม่พบ $REPO_DIR"
run_logged "eval_${MODEL}_seed${SEED}" python3 experiments/evaluate.py --model_h5 "$H5" --model "$LABEL" --seed "$SEED" \
    --lab_csv "$LAB_TEST_CSV" --field_csv "$FIELD_TEST_CSV" --results "$RESULTS_CSV" --pred_dir "$PRED_DIR" --gpu "$GPU"
