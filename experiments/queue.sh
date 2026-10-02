#!/usr/bin/env bash
# Job queue for one GPU: runs the jobs in a queue file one after another (run it inside screen).
# usage: bash experiments/queue.sh <gpu> <queue_file>
#
# queue file — one job per line (edit any time; lines starting with # are skipped):
#   c2_pretrain                 C2 pre-train (R1 -> R2) on mini-ImageNet
#   downstream <model> <seed>   run_downstream.sh, then run_eval.sh (lab + field) automatically
#   eval <model> <seed>         run_eval.sh only
# finished lines become "# DONE <time> ...", failed lines "# FAIL <time> ...".
# A job that is not ready yet (C2 before pre-train finished, C1 before E_C1 is set) is skipped and retried later.
# When nothing is runnable the queue sleeps 10 min and re-reads the file, so jobs can be appended while it runs.
set -uo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
source "$EXP/_common.sh"

GPU=${1:?usage: queue.sh <gpu> <queue_file>}
QF=${2:?queue file}
[[ -f $QF ]] || die "ไม่พบ queue file: $QF"
C2_PRE_H5=$SAVE_DIR/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5

gpu_busy() { [[ -n $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; }

ready() {  # ready <job...> : 0 = can run now
    case $1 in
        downstream|eval)
            source "$CONFIG_FILE"   # re-read: E_C1 may have been set meanwhile
            [[ $2 == C1 && "$E_C1" == *__SET_ME__* ]] && return 1
            [[ $2 == C2 && ! -f $C2_PRE_H5 ]] && return 1
            return 0 ;;
        c2_pretrain) return 0 ;;
        *) return 0 ;;
    esac
}

run_job() {
    case $1 in
        c2_pretrain) bash "$EXP/run_C2_pretrain.sh" "$GPU" ;;
        downstream)  bash "$EXP/run_downstream.sh" "$2" "$3" "$GPU" && bash "$EXP/run_eval.sh" "$2" "$3" "$GPU" ;;
        eval)        bash "$EXP/run_eval.sh" "$2" "$3" "$GPU" ;;
        *) echo "[queue] ไม่รู้จัก job: $*"; return 2 ;;
    esac
}

mark() {  # mark <line_no> <DONE|FAIL>  (portable, no sed -i)
    local tag="# $2 $(date '+%m-%d %H:%M') " tmp
    tmp=$(mktemp) && awk -v n="$1" -v t="$tag" 'NR==n {$0 = t $0} {print}' "$QF" > "$tmp" && cat "$tmp" > "$QF" && rm -f "$tmp" \
        || die "mark ไม่สำเร็จ (line $1) — หยุด queue เพื่อไม่ให้รันงานซ้ำ"
}

info "queue GPU $GPU <- $QF"
while true; do
    picked=""
    n=0
    while IFS= read -r line || [[ -n $line ]]; do
        n=$((n + 1))
        line=${line%%#*}; read -r -a job <<< "$line"
        [[ ${#job[@]} -eq 0 ]] && continue
        if ready "${job[@]}"; then picked=$n; break; fi
    done < "$QF"

    if [[ -z $picked ]]; then
        sleep 600; continue
    fi
    while gpu_busy; do info "GPU $GPU ยังมีงานอื่นอยู่ — รอ 10 นาที"; sleep 600; done

    line=$(sed -n "${picked}p" "$QF"); line=${line%%#*}; read -r -a job <<< "$line"
    info "[queue GPU $GPU] start: ${job[*]}  ($(date '+%m-%d %H:%M'))"
    if run_job "${job[@]}"; then mark "$picked" DONE; else mark "$picked" FAIL; fi
done
