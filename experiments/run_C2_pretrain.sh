#!/usr/bin/env bash
# X1-C2 pre-train: same-image vs different-image on mini-ImageNet (no flip), EfficientNet-B5
#   R1 transfer (FC)  ->  R2 unfreeze block4-block7   (same settings as the unlearned model in the paper)
# usage: bash experiments/run_C2_pretrain.sh <gpu> [r1|r2]     (default: r1 then r2)
set -uo pipefail
source "$(dirname "$0")/_common.sh"

GPU=${1:?usage: run_C2_pretrain.sh <gpu> [r1|r2]}
STAGE=${2:-all}

require_vars SAVE_DIR MINI_CSV E_UN_R1 E_UN_R2
require_paths MINI_CSV
activate_env
cd "$REPO_DIR/CNNs_unlearn" || die "ไม่พบ $REPO_DIR/CNNs_unlearn"

BASE=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet
R1_CKPT=$BASE/R1/transfer/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_transfer-R1.h5

if [[ $STAGE == all || $STAGE == r1 ]]; then
    run_logged C2_pretrain_R1 python3 trainmodel.py --gpu "$GPU" --network_name EffNetB5 --weight imagenet \
        --set baseML_unlearn --task sameimg --data_path "$MINI_CSV" --save_dir "$SAVE_DIR" \
        --name transfer --R 1 --epochs "$E_UN_R1"
fi

if [[ $STAGE == all || $STAGE == r2 ]]; then
    require_paths R1_CKPT
    run_logged C2_pretrain_R2 python3 trainmodel.py --gpu "$GPU" --network_name EffNetB5 --weight imagenet \
        --set baseML_unlearn --task sameimg --data_path "$MINI_CSV" --save_dir "$SAVE_DIR" \
        --name unfreezeB4-B7 --R 2 --epochs "$E_UN_R2" --checkpoint_dir "$R1_CKPT"
fi

info "C2 pre-train model: $BASE/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5"
