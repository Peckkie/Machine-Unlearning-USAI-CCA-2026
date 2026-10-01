#!/usr/bin/env bash
# สรุปสถานะทุก run ของ X1 / X2 จาก log, โมเดลที่เซฟ และ results_seeds.csv
# usage: bash experiments/status.sh
source "$(dirname "$0")/_common.sh"

# progress LOGNAME : "epoch N/M  val_acc X" from a Keras log
progress() {
    local f=$LOG_DIR/$1.log
    [[ -f $f ]] || { echo "-"; return; }
    local ep va
    ep=$(grep -ao "Epoch [0-9]*/[0-9]*" "$f" | tail -1 | cut -d' ' -f2)
    va=$(grep -ao "val_accuracy: [0-9.]*" "$f" | tail -1 | cut -d' ' -f2)
    if grep -q "Traceback (most recent call last)" "$f" && ! grep -q "Saved model to disk" "$f"; then
        echo "ERROR (ep ${ep:-?}) — tail $f"
    else
        echo "ep ${ep:-0}  val_acc ${va:--}"
    fi
}

result() {  # result MODEL SEED TESTSET -> accuracy
    [[ -f $RESULTS_CSV ]] || { echo "-"; return; }
    awk -F, -v m="$1" -v s="$2" -v t="$3" 'NR>1 && $1==m && $2==s && $3==t {printf "%.3f", $4; f=1} END {if (!f) printf "-"}' "$RESULTS_CSV"
}

line() { printf "%-14s %-5s %-34s %-34s %-6s %-6s\n" "$@"; }

echo "== screens =="; screen -ls 2>/dev/null | grep -E "^\s+[0-9]+\." | sed 's/^/  /'
echo "== GPU =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
echo
echo "== C2 pre-train (mini-ImageNet, same/diff image) =="
C2_H5=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5
echo "  R1: $(progress C2_pretrain_R1)"
echo "  R2: $(progress C2_pretrain_R2)"
echo "  model: $([[ -f $C2_H5 ]] && echo DONE || echo -)"
echo
echo "== downstream (USAI 15AB) =="
line MODEL SEED "R1" "R2" LAB FIELD
for run in original:1 original:2 original:3 unlearned:1 unlearned:2 unlearned:3 C1:1 C1:2 C1:3 C2:1 C2:2 C2:3; do
    m=${run%%:*}; s=${run##*:}
    if [[ $m == unlearned && $s == 1 ]]; then
        line "$m" "$s" "(paper model)" "(paper model)" "$(result "$m" "$s" lab)" "$(result "$m" "$s" field)"
        continue
    fi
    setup_model "$m" "$s"
    r2=$(progress "${m}_seed${s}_R2"); [[ -f $R2_DIR/$R2_NAME.h5 ]] && r2="DONE  $r2"
    line "$m" "$s" "$(progress "${m}_seed${s}_R1")" "$r2" "$(result "$m" "$s" lab)" "$(result "$m" "$s" field)"
done
[[ -f $LOG_DIR/original_seed99_R1.log ]] && { echo; echo "== smoke test (original seed 99: R1 2 ep, R2 3 ep) =="
    echo "  R1: $(progress original_seed99_R1)"; echo "  R2: $(progress original_seed99_R2)"; }
