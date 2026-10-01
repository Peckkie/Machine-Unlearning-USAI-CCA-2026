#!/usr/bin/env bash
# ตรวจ config_m29.sh ก่อนเทรน: path ทุกตัว, ภาพใน CSV, conda env, GPU, จำนวนภาพ train
# usage: bash experiments/check_setup.sh      (ไม่เทรนอะไร แค่ตรวจ — ส่ง output ทั้งหมดกลับมาได้เลย)
source "$(dirname "$0")/_common.sh"

ok()   { echo -e "  \033[32m[OK]\033[0m   $*"; }
bad()  { echo -e "  \033[31m[MISS]\033[0m $*"; }
chk()  { local v=$1; if [[ "${!v}" == *__SET_ME__* ]]; then bad "$v ยังไม่ได้ตั้งค่า"; elif [[ -e "${!v}" ]]; then ok "$v = ${!v}"; else bad "$v = ${!v}"; fi; }

echo "== paths =="
for v in REPO_DIR USAI_DATA_DIR MINI_CSV LAB_TEST_CSV FIELD_TEST_CSV PAPER_UNLEARNED_H5 PAPER_ORIGINAL_H5 \
         UNLEARN_B4B7_CKPT TB_UNLEARN_R1 TB_UNLEARN_R2; do chk "$v"; done
for f in Traindf_fold4_8_v1.csv Valdf_fold3_v1.csv; do
    [[ -f $USAI_DATA_DIR/$f ]] && ok "$USAI_DATA_DIR/$f" || bad "$USAI_DATA_DIR/$f"
done
[[ "$SAVE_DIR" == *__SET_ME__* ]] && bad "SAVE_DIR ยังไม่ได้ตั้งค่า" || {
    mkdir -p "$SAVE_DIR" 2>/dev/null && [[ -w $SAVE_DIR ]] && ok "SAVE_DIR เขียนได้: $SAVE_DIR" || bad "SAVE_DIR เขียนไม่ได้: $SAVE_DIR"; }

echo "== conda env: ${CONDA_ENV} =="
if [[ "$CONDA_ENV" == *__SET_ME__* ]]; then
    bad "CONDA_ENV ยังไม่ได้ตั้งค่า — env ที่มี:"; conda env list 2>/dev/null | sed 's/^/         /'
else
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || bad "activate $CONDA_ENV ไม่ได้"
    for m in tensorflow efficientnet.tfkeras skimage keras.utils.generic_utils pandas sklearn scipy; do
        python3 -c "import $m" 2>/dev/null && ok "import $m" || bad "import $m"
    done
    python3 -c "import tensorflow as tf; print('  TF', tf.__version__, '| GPUs:', [d.name for d in tf.config.list_physical_devices('GPU')])" 2>/dev/null
fi

echo "== data =="
python3 - "$USAI_DATA_DIR" "$MINI_CSV" "$LAB_TEST_CSV" "$FIELD_TEST_CSV" "${MINI_PATH_REPLACE:-}" <<'EOF' 2>&1 | sed 's/^/  /'
import os, sys
try:
    import pandas as pd
except ImportError:
    print('[MISS] pandas ไม่มีใน env นี้'); sys.exit()
data_dir, mini, lab, field, rep = sys.argv[1:6]
def sample(name, csv, col, filt=None, replace=''):
    if not os.path.isfile(csv):
        print(f'[MISS] {name}: ไม่พบ {csv}'); return
    d = pd.read_csv(csv, dtype=str)
    if filt: d = filt(d)
    if col not in d.columns:
        print(f'[MISS] {name}: ไม่มีคอลัมน์ {col!r} (มี {list(d.columns)[:12]})'); return
    paths = d[col].dropna()
    if replace:
        old, new = replace.split('=', 1); paths = paths.str.replace(old, new, n=1, regex=False)
    s = paths.sample(min(20, len(paths)), random_state=0)
    n_ok = sum(os.path.isfile(p) for p in s)
    print(f'[{"OK" if n_ok == len(s) else "MISS"}] {name}: {len(d)} rows, ภาพที่สุ่ม {n_ok}/{len(s)} มีจริง  ตัวอย่าง: {s.iloc[0]}')
for f in ['Traindf_fold4_8_v1.csv', 'Valdf_fold3_v1.csv']:
    sample(f, f'{data_dir}/{f}', 'Path Crop')
sample('mini-ImageNet', mini, 'img_path', replace=rep)
sample('Lab test', lab, 'Path Crop', lambda d: d[(d['Path Crop'] != 'None') & (d['Path Crop'] != 'Nan')])
sample('Field test', field, 'Path Crop_I7', lambda d: d[d['Sub_Class_15AB'] != 'None'])
EOF
echo "== done =="
