#!/usr/bin/env bash
# หา conda env ที่ใช้เทรนได้: มี tensorflow + efficientnet + scikit-image + keras + pandas + sklearn + scipy และ "มองเห็น GPU"
# usage: bash experiments/probe_envs.sh                 # ทุก env
#        bash experiments/probe_envs.sh base bitnetenv2  # เฉพาะ env ที่ระบุ (ชื่อ หรือ path)
envs=()
if [[ $# -gt 0 ]]; then
    for n in "$@"; do
        for e in "$n" ~/miniconda3/envs/"$n" ~/anaconda3/envs/"$n" /home/kannika/miniconda3/envs/"$n"; do
            [[ $n == base ]] && e=~/miniconda3
            [[ -x "$e/bin/python" ]] && { envs+=("$e"); break; }
        done
    done
else
    envs=(~/miniconda3 ~/miniconda3/envs/* ~/anaconda3 ~/anaconda3/envs/* /home/kannika/miniconda3/envs/* /opt/conda/envs/*)
fi
for e in "${envs[@]}"; do
    [[ -x "$e/bin/python" ]] || continue
    out=$("$e/bin/python" -c "
import tensorflow as t, efficientnet.tfkeras, skimage, keras.utils.generic_utils, pandas, sklearn, scipy, sys
g = t.config.list_physical_devices('GPU')
sys.stdout.write(('READY' if g else 'NO-GPU') + ' tf ' + t.__version__ + ' gpus=' + str(len(g)))" 2>&1 | tail -1)
    echo "$e => $out"
done
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
