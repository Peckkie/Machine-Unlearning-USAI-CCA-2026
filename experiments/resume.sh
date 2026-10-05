#!/usr/bin/env bash
# Resume an interrupted stage (power cut, crash) from its newest per-epoch checkpoint, training only the remaining epochs.
# usage: bash experiments/resume.sh downstream <model> <seed> <gpu> <R1|R2>
#        bash experiments/resume.sh c2_pretrain <gpu> <R1|R2>
# completed epochs = epochs logged in TensorBoard (all runs of that stage); checkpoint = newest on_epoch_end*/..._last.h5 or _epochN.h5
# NOTE: train.py / trainmodel.py re-compile after loading, so the optimizer state restarts (same as the original paper's resumes).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

KIND=${1:?usage: resume.sh downstream <model> <seed> <gpu> <R1|R2> | c2_pretrain <gpu> <R1|R2>}
if [[ $KIND == downstream ]]; then
    MODEL=${2:?model}; SEED=${3:?seed}; GPU=${4:?gpu}; STAGE=${5:?R1|R2}
    setup_model "$MODEL" "$SEED"
    if [[ $STAGE == R1 ]]; then ROOT=$(dirname "$R1_DIR"); TOTAL=$E_DS_R1; else ROOT=$(dirname "$R2_DIR"); TOTAL=$R2_EPOCHS; fi
else
    GPU=${2:?gpu}; STAGE=${3:?R1|R2}
    BASE=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet
    if [[ $STAGE == R1 ]]; then ROOT=$BASE/R1/transfer; TOTAL=$E_UN_R1; else ROOT=$BASE/R2/unfreezeB4-B7; TOTAL=$E_UN_R2; fi
fi
[[ -d $ROOT ]] || die "ไม่พบโฟลเดอร์ของ stage นี้: $ROOT"

activate_env
check_env
DONE=$(cd "$EXP_DIR" && python3 -c "from compute_budget import tb_wall_time; print(tb_wall_time('$ROOT/Mylogs_tensor')[1])" 2>/dev/null) \
    || die "อ่านจำนวน epoch จาก TensorBoard ไม่ได้: $ROOT/Mylogs_tensor"
CKPT=$(ls -t "$ROOT"/on_epoch_end*/*.h5 2>/dev/null | grep -v '\.weights\.h5$' | head -1)
[[ -n $CKPT ]] || die "ไม่พบ checkpoint ใน $ROOT/on_epoch_end*"
LEFT=$((TOTAL - DONE))
N_RES=$(ls -d "$ROOT"/on_epoch_end_resume* 2>/dev/null | wc -l)
EPNAME=on_epoch_end_resume$((N_RES + 1))
info "resume $KIND ${MODEL:-} ${SEED:-} $STAGE: เสร็จแล้ว $DONE/$TOTAL epoch -> เทรนต่ออีก $LEFT epoch"
echo "  checkpoint: $CKPT ($(date -r "$CKPT" '+%m-%d %H:%M'))"
[[ $LEFT -gt 0 ]] || die "stage นี้ครบ $TOTAL epoch แล้ว (ถ้ายังไม่มีโมเดลใน models/ ให้ใช้ checkpoint ล่าสุดแทน)"

if [[ $KIND == downstream ]]; then
    EXP_ARGS=(); [[ -n $EXP ]] && EXP_ARGS=(--exp "$EXP")
    TAG_ARGS=(); [[ -n $TAG ]] && TAG_ARGS=(--tag "$TAG")
    [[ -n ${USAI_PATH_REPLACE:-} ]] && TAG_ARGS+=(--path_replace "$USAI_PATH_REPLACE")
    if [[ $STAGE == R1 ]]; then NAME=transfer; R=1; LR=$LR_DS_R1; else NAME=$DS_R2_NAME; R=2; LR=$LR_DS_R2; fi
    cd "$REPO_DIR/USAI_unlearn" || die "no USAI_unlearn"
    run_logged "${MODEL}_seed${SEED}_${STAGE}" python3 train.py --gpu "$GPU" --network_name EffNetB5 --weight imagenet --set "$SET" \
        ${TAG_ARGS[@]+"${TAG_ARGS[@]}"} ${EXP_ARGS[@]+"${EXP_ARGS[@]}"} --data_path "$USAI_DATA_DIR" --save_dir "$SAVE_DIR" --data unbalanced \
        --batchsize "$BATCH_SIZE" --seed "$SEED" --effnet_impl "$EFFNET_IMPL" --name "$NAME" --R "$R" --lr "$LR" \
        --resume --checkpoint_dir "$CKPT" --epochs "$LEFT" --epochendName "$EPNAME"
else
    PR_ARGS=(); [[ -n ${MINI_PATH_REPLACE:-} ]] && PR_ARGS=(--path_replace "$MINI_PATH_REPLACE")
    if [[ $STAGE == R1 ]]; then NAME=transfer; R=1; LR=$LR_UN_R1; else NAME=unfreezeB4-B7; R=2; LR=$LR_UN_R2; fi
    cd "$REPO_DIR/CNNs_unlearn" || die "no CNNs_unlearn"
    run_logged "C2_pretrain_${STAGE}" python3 trainmodel.py --gpu "$GPU" --network_name EffNetB5 --weight imagenet \
        --set baseML_unlearn --task sameimg --data_path "$MINI_CSV" --save_dir "$SAVE_DIR" --name "$NAME" --R "$R" \
        --lr "$LR" --batchsize "$BATCH_SIZE_UN" ${PR_ARGS[@]+"${PR_ARGS[@]}"} \
        --resume --checkpoint_dir "$CKPT" --epochs "$LEFT" --epochendName "$EPNAME"
fi
info "resume เสร็จ: $KIND ${MODEL:-} ${SEED:-} $STAGE"
