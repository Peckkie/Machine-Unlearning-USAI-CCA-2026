#!/usr/bin/env bash
# shared helpers for run_*.sh — load config, check paths, activate conda

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/config_m29.sh"

die() { echo -e "\033[31m[ERROR] $*\033[0m" >&2; exit 1; }
info() { echo -e "\033[32m[INFO] $*\033[0m"; }

# require_vars VAR1 VAR2 ... : must be set and not __SET_ME__
require_vars() {
    for v in "$@"; do
        [[ -z "${!v:-}" || "${!v}" == *__SET_ME__* ]] && die "ยังไม่ได้ตั้งค่า $v ใน experiments/config_m29.sh"
    done
}

# require_paths VAR1 VAR2 ... : file/folder must exist
require_paths() {
    for v in "$@"; do
        [[ -e "${!v}" ]] || die "$v = ${!v} ไม่มีอยู่จริง"
    done
}

activate_env() {
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
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
