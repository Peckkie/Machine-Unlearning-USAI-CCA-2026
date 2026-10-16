#!/usr/bin/env bash
# Evaluate the legacy single-run models behind Table 4/5 of the submitted paper with the same evaluate.py
# (lab 1312 + field 807), saved as <label> seed 1 in $RESULTS_CSV / $PRED_DIR.  Machine 29.
# usage: bash experiments/run_eval_legacy.sh <gpu>
#   ViT models need vit_keras -> probabilities from $VIT_PY (predict_probs.py), metrics in the AI env (evaluate.py --probs_dir)
set -uo pipefail
source "$(dirname "$0")/_common.sh"
GPU=${1:?usage: run_eval_legacy.sh <gpu>}
require_vars SAVE_DIR
require_paths LAB_TEST_CSV FIELD_TEST_CSV
VIT_PY=${VIT_PY:-/home/kannika/miniconda3/envs/vit-tf-clean/bin/python}
M=/media/tohn/HDD2/Model_unlearn
L=$SAVE_DIR/legacy_models
# ResNet unlearned conv3 / conv1-conv3: downstream conv3-conv5 models copied from machine 28 (exp_unfreeze_conv3_block-conv5_block/exp_*)
# label | model file  (unlearned B4-B7 = paper model = "unlearned seed 1", already evaluated)
MODELS=(
  "legacy_effnet_original|/media/tohn/SSD/ModelTrainByImages/R2_1/models/B5R2_block5_15AB_1FC_3.h5"
  "legacy_effnet_unl_B4|$M/EffNetB5Model/MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4/models/modelEffNetB5_MLunlearn_USAI_unfreezeBlock5a_se_excite_exp_unfreezeB4-R2_unbalanced.h5"
  "legacy_effnet_unl_B1B4|$L/modelEffNetB5_MLunlearn_USAI_unfreezeBlock5a_se_excite_exp_unfreezeB1-B4-R2_unbalanced.h5"
  "legacy_effnet_unl_B5B7|$M/Models_USAI/ML-unlearn/R2/unfreezeBlock5a_se_excite/models/modelEffNetB5_ML-unlearn_unfreezeBlock5a_se_excite-R2.h5"
  "legacy_resnet_original|$M/ResNet152v2Model/MLorigin_USAI/R2_unbalanced/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_MLorigin_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.h5"
  "legacy_resnet_unl_c3|$L/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv3_block-R2_unbalanced.h5"
  "legacy_resnet_unl_c3c5|$M/ResNet152v2Model/MLunlearn_USAI/R2_unbalanced/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block-R2_unbalanced.h5"
  "legacy_resnet_unl_c1c3|$L/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv1-conv3_block-R2_unbalanced.h5"
  "legacy_vit_original|$M/Models_USAI/ViTModel/ViTorigin_USAI/R1_unbalanced/exp1/models/modelVITL32_USAI15ab-R1_unbalanced.h5"
  "legacy_vit_unlearned|$M/ViTModel_unlearn/ViTunlearn_USAI/R1_unbalanced/exp1/models/modelVITL32_Unlearning_USAI15ab-R1_unbalanced.h5"
)
activate_env
check_env
cd "$REPO_DIR" || die "ไม่พบ $REPO_DIR"
EV=(--lab_csv "$LAB_TEST_CSV" --field_csv "$FIELD_TEST_CSV" --results "$RESULTS_CSV" --pred_dir "$PRED_DIR" --gpu "$GPU")
PROBS=$SAVE_DIR/legacy_probs
python3 experiments/evaluate.py --model x --seed 1 --lab_csv "$LAB_TEST_CSV" --field_csv "$FIELD_TEST_CSV" --dump_paths "$PROBS"
for item in "${MODELS[@]}"; do
    label=${item%%|*}; h5=${item#*|}
    [[ -f $h5 ]] || { info "ไม่พบ $h5 -> ข้าม $label"; continue; }
    if [[ $label == legacy_vit_* ]]; then
        for ts in lab field; do
            ( run_logged "probs_${label}_$ts" "$VIT_PY" experiments/predict_probs.py --model_h5 "$h5" \
                --paths "$PROBS/${ts}_paths.txt" --out "$PROBS/${label}_seed1_$ts.npy" --gpu "$GPU" ) || info "FAILED probs $label $ts"
        done
        ( run_logged "eval_$label" python3 experiments/evaluate.py --model "$label" --seed 1 --probs_dir "$PROBS" "${EV[@]}" ) || info "FAILED $label"
    else
        ( run_logged "eval_$label" python3 experiments/evaluate.py --model_h5 "$h5" --model "$label" --seed 1 "${EV[@]}" ) || info "FAILED $label"
    fi
done
info "legacy evaluation done -> $RESULTS_CSV"
