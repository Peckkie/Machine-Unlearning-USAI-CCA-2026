#!/usr/bin/env bash
# หา conda env ที่ใช้เทรนได้ (มี tensorflow + efficientnet + scikit-image + keras + pandas + sklearn + scipy) และดู GPU
# usage: bash experiments/probe_envs.sh
for e in ~/miniconda3 ~/miniconda3/envs/* ~/anaconda3 ~/anaconda3/envs/* /home/kannika/miniconda3/envs/* /opt/conda/envs/*; do
    [[ -x "$e/bin/python" ]] || continue
    out=$("$e/bin/python" -c "import tensorflow as t, efficientnet.tfkeras, skimage, keras.utils.generic_utils, pandas, sklearn, scipy; import sys; sys.stdout.write('READY tf ' + t.__version__)" 2>&1 | tail -1)
    echo "$e => $out"
done
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
