#!/usr/bin/env bash
# X1 / X2 downstream on USAI 15AB (EfficientNet-B5, split เดียวกับ paper: USAI_unlearn/train.py)
#   R1 transfer (FC)  ->  R2 unfreeze Block5a_se_excite-Block7
#
# usage: bash experiments/run_downstream.sh <model> <seed> <gpu>
#   model : original   ImageNet -> USAI                                (X2)
#           unlearned  ImageNet -> flip unlearn (B4-B7) -> USAI        (X2)
#           C1         ImageNet -> USAI, R2 = E_C1 epochs (compute-matched)
#           C2         ImageNet -> same/diff-image (B4-B7) -> USAI    (run_C2_pretrain.sh ก่อน)
#   seed  : 1 2 3
# R1 ที่เทรนเสร็จแล้วจะถูกข้าม (รันซ้ำได้ถ้า R2 ค้าง)
set -uo pipefail
source "$(dirname "$0")/_common.sh"

MODEL=${1:?usage: run_downstream.sh <original|unlearned|C1|C2> <seed> <gpu>}
SEED=${2:?seed}
GPU=${3:?gpu}

require_vars SAVE_DIR BATCH_SIZE
require_paths USAI_DATA_DIR

R2_EPOCHS=$E_DS_R2
case $MODEL in
    original)  SET=MLorigin_USAI;  TAG="" ;;
    C1)        SET=MLorigin_USAI;  TAG=C1_computematched; require_vars E_C1; R2_EPOCHS=$E_C1 ;;
    unlearned) SET=MLunlearn_USAI; TAG="";         EXP=unfreezeB4-B7
               require_vars UNLEARN_B4B7_CKPT; PRE_CKPT=$UNLEARN_B4B7_CKPT ;;
    C2)        SET=MLunlearn_USAI; TAG=C2_sameimg; EXP=unfreezeB4-B7
               PRE_CKPT=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5 ;;
    *) die "model ต้องเป็น original | unlearned | C1 | C2" ;;
esac
[[ -n ${PRE_CKPT:-} ]] && require_paths PRE_CKPT

## same naming rule as USAI_unlearn/train.py
SET_DIR=$SET${TAG:+_$TAG}
RUN_TAG=${TAG:+_$TAG}_seed$SEED
if [[ $SET == MLorigin_USAI ]]; then
    R1_DIR=$SAVE_DIR/EffNetB5Model/$SET_DIR/weight_imagenet/R1_unbalanced/transfer/seed$SEED/models
    R1_NAME=modelEffNetB5_${SET}_transfer-R1_unbalanced$RUN_TAG
    EXP_ARGS=()
    PRE_ARGS=()
else
    R1_DIR=$SAVE_DIR/EffNetB5Model/$SET_DIR/R1_unbalanced/transfer_exp_$EXP/seed$SEED/models
    R1_NAME=modelEffNetB5_${SET}_transfer_exp_$EXP-R1_unbalanced$RUN_TAG
    EXP_ARGS=(--exp "$EXP")
    PRE_ARGS=(--checkpoint_dir "$PRE_CKPT")
fi
TAG_ARGS=(); [[ -n $TAG ]] && TAG_ARGS=(--tag "$TAG")
COMMON=(--gpu "$GPU" --network_name EffNetB5 --weight imagenet --set "$SET" ${TAG_ARGS[@]+"${TAG_ARGS[@]}"} ${EXP_ARGS[@]+"${EXP_ARGS[@]}"}
        --data_path "$USAI_DATA_DIR" --save_dir "$SAVE_DIR" --data unbalanced --batchsize "$BATCH_SIZE" --seed "$SEED")

activate_env
cd "$REPO_DIR/USAI_unlearn" || die "ไม่พบ $REPO_DIR/USAI_unlearn"

## ---- R1 transfer (FC) ----
if [[ -f $R1_DIR/$R1_NAME.json && -f $R1_DIR/$R1_NAME.weights.h5 ]]; then
    info "R1 เสร็จแล้ว ข้าม: $R1_DIR/$R1_NAME"
else
    run_logged "${MODEL}_seed${SEED}_R1" python3 train.py "${COMMON[@]}" ${PRE_ARGS[@]+"${PRE_ARGS[@]}"} \
        --name transfer --R 1 --epochs "$E_DS_R1" --lr "$LR_DS_R1"
fi

## ---- R2 fine-tune Block5a_se_excite-Block7 ----
run_logged "${MODEL}_seed${SEED}_R2" python3 train.py "${COMMON[@]}" \
    --name unfreezeBlock5a_se_excite --R 2 --epochs "$R2_EPOCHS" --lr "$LR_DS_R2" \
    --checkpoint_dir "$R1_DIR/$R1_NAME.weights.h5" --Modeljson_dir "$R1_DIR/$R1_NAME.json"

info "เสร็จ: $MODEL seed $SEED"
