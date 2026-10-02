#!/usr/bin/env bash
# shared helpers for run_*.sh — load config, check paths, activate conda

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# which machine: env MACHINE, or experiments/.machine (set once per machine: echo 28 > experiments/.machine), default 29
MACHINE=${MACHINE:-$(cat "$EXP_DIR/.machine" 2>/dev/null || echo 29)}
CONFIG_FILE="$EXP_DIR/config_m${MACHINE}.sh"
[[ -f $CONFIG_FILE ]] || { echo "[ERROR] ไม่พบ $CONFIG_FILE" >&2; exit 1; }
source "$CONFIG_FILE"

die() { echo -e "\033[31m[ERROR] $*\033[0m" >&2; exit 1; }
info() { echo -e "\033[32m[INFO] $*\033[0m"; }

# require_vars VAR1 VAR2 ... : must be set and not __SET_ME__
require_vars() {
    for v in "$@"; do
        [[ -z "${!v:-}" || "${!v}" == *__SET_ME__* ]] && die "ยังไม่ได้ตั้งค่า $v ใน experiments/config_m${MACHINE}.sh"
    done
}

# require_paths VAR1 VAR2 ... : file/folder must exist
require_paths() {
    for v in "$@"; do
        [[ -e "${!v}" ]] || die "$v = ${!v} ไม่มีอยู่จริง"
    done
}

# conda.sh: from PATH, or the usual install folders (non-interactive ssh does not load conda)
source_conda() {
    local base
    base=$(conda info --base 2>/dev/null)
    for b in "$base" ~/miniconda3 ~/anaconda3 /opt/conda; do
        # shellcheck disable=SC1090,SC1091
        [[ -n $b && -f $b/etc/profile.d/conda.sh ]] && { source "$b/etc/profile.d/conda.sh"; return 0; }
    done
    die "หา conda ไม่เจอ"
}

activate_env() {
    require_vars CONDA_ENV
    source_conda
    # conda activate scripts (e.g. AI env: env_vars.sh uses $LD_LIBRARY_PATH) break under `set -u`
    local had_u=0; [[ $- == *u* ]] && had_u=1
    set +u
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
    conda activate "$CONDA_ENV" || die "conda activate $CONDA_ENV ไม่ได้"
    [[ $had_u == 1 ]] && set -u
    return 0
}

# run_logged NAME cmd... : print the command, run it, keep a log in $LOG_DIR/NAME.log
run_logged() {
    local name=$1; shift
    mkdir -p "$LOG_DIR"
    info "$name"
    echo "  $*"
    "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "$name ล้มเหลว ดู log: $LOG_DIR/$name.log"
}

# check that the conda env has everything train.py / trainmodel.py import
check_env() {
    python3 -c "import tensorflow, efficientnet.tfkeras, skimage, keras.utils.generic_utils, pandas, sklearn" 2>/dev/null \
        || die "conda env '$CONDA_ENV' ขาด package (ต้องมี tensorflow, efficientnet, scikit-image, keras, pandas, scikit-learn)"
}

# setup_model MODEL SEED : set SET TAG EXP PRE_CKPT R2_EPOCHS EFFNET_IMPL R1_DIR R1_NAME R2_DIR R2_NAME
#   naming rule = USAI_unlearn/train.py
setup_model() {
    MODEL=$1; SEED=$2
    R2_EPOCHS=$E_DS_R2; EXP=""; PRE_CKPT=""; EFFNET_IMPL=efn
    case $MODEL in
        original)  SET=MLorigin_USAI;  TAG="" ;;
        C1)        SET=MLorigin_USAI;  TAG=C1_computematched; R2_EPOCHS=${E_C1} ;;
        unlearned) SET=MLunlearn_USAI; TAG="";         EXP=unfreezeB4-B7; PRE_CKPT=$UNLEARN_B4B7_CKPT ;;
        C2)        SET=MLunlearn_USAI; TAG=C2_sameimg; EXP=unfreezeB4-B7
                   PRE_CKPT=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5 ;;
        *) die "model ต้องเป็น original | unlearned | C1 | C2" ;;
    esac
    local set_dir=$SET${TAG:+_$TAG}
    local run_tag=${TAG:+_$TAG}_seed$SEED
    local base=$SAVE_DIR/EffNetB5Model/$set_dir
    if [[ $SET == MLorigin_USAI ]]; then
        R1_DIR=$base/weight_imagenet/R1_unbalanced/transfer/seed$SEED/models
        R1_NAME=modelEffNetB5_${SET}_transfer-R1_unbalanced$run_tag
        R2_DIR=$base/weight_imagenet/R2_unbalanced/$DS_R2_NAME/seed$SEED/models
        R2_NAME=modelEffNetB5_${SET}_$DS_R2_NAME-R2_unbalanced$run_tag
    else
        R1_DIR=$base/R1_unbalanced/transfer_exp_$EXP/seed$SEED/models
        R1_NAME=modelEffNetB5_${SET}_transfer_exp_$EXP-R1_unbalanced$run_tag
        R2_DIR=$base/R2_unbalanced/$DS_R2_NAME/exp_$EXP/seed$SEED/models
        R2_NAME=modelEffNetB5_${SET}_${DS_R2_NAME}_exp_$EXP-R2_unbalanced$run_tag
    fi
}
