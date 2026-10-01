# X1 / X2 — การทดลองเพิ่มเติมตอบกรรมการ (EfficientNet-B5)

| งาน | ตอบกรรมการ | รันใหม่ |
|---|---|---|
| X1-C1 ordinary fine-tuning (compute-matched) | ข้อ 2, 11 | downstream 3 seeds |
| X1-C2 unrelated auxiliary (same-image vs different-image) | ข้อ 2, 11 | pre-train 1 ครั้ง + downstream 3 seeds |
| X1-C3 pre-train แบบไม่มี flip augmentation | ข้อ 2, 11 | ไม่รัน → เขียนใน Limitations |
| X2 repeated runs 3 seeds (original, unlearned, C1, C2) | ข้อ 5 | original 2 + unlearned 2 seeds |

**รวม EfficientNet-B5 = 12 รอบใหม่** = C2 pre-train 1 + original ×3 + unlearned ×2 + C1 ×3 + C2 ×3

- unlearned seed 1 = โมเดลเดิมใน paper (มาจาก `train.py` pipeline เดียวกัน) → รันเพิ่ม seed 2, 3
- original ต้องรันใหม่ **ครบ 3 seeds** เพราะ original ใน paper เป็นโมเดล legacy (`ModelTrainByImages/R2_1/...B5R2_block5_15AB_1FC_3.h5`)
  คนละ pipeline กับ `train.py` และ original ที่เคยเทรนด้วย `train.py` พัง (ดูข้อ ⚠️ ด้านล่าง)

> ⚠️ ทุก run ใช้ split เดียวกับผลใน paper (`train.py`) และ test set ชุดเดิม: Lab Testset1312 + Field Testset807

---

## สิ่งที่เพิ่มในโค้ด

| ไฟล์ | เพิ่ม |
|---|---|
| `CNNs_unlearn/data_generator.py` | `SameDiff_generator` — pair เดิม label เดิม (`cls`) augmentation เดิม แต่ label 0 = **ภาพอื่น** แทนภาพ flip (ไม่มี flip เลย) |
| `CNNs_unlearn/trainmodel.py` | `--task [flip, sameimg]` → `sameimg` เซฟใน `baseML_unlearn_sameimg/` |
| `USAI_unlearn/train.py`, `train-Kfold.py` | `--seed` (ตั้ง seed python/numpy/tf + data generator) และ `--tag` (เซฟใน `{set}_{tag}/`) |
| `experiments/config_m29.sh` | path / ค่าทั้งหมดของเครื่อง 29 (แก้ไฟล์นี้ไฟล์เดียว) |
| `experiments/check_setup.sh` | ตรวจ path / ภาพ / env / GPU ก่อนเทรน |
| `experiments/run_C2_pretrain.sh`, `run_downstream.sh` | script รันพร้อมเช็ค path + log |
| `experiments/evaluate.py`, `run_eval.sh` | evaluate lab (1312) + field (807) แบบเดียวกับ notebook ใน paper → `results_seeds.csv` |
| `USAI_unlearn/EffNetmodels.py` | `--effnet_impl efn` แก้ bug original baseline (ดูด้านล่าง) |
| `experiments/compute_budget.py` | คำนวณจำนวน epoch ที่ต้องเพิ่มให้ C1 |
| `experiments/seed_stats.py` | mean ± SD + paired t-test ข้าม seeds |

ไม่ใส่ `--seed` / `--tag` / `--task` = ทำงานเหมือนเดิมทุกอย่าง (path เดิม ชื่อไฟล์เดิม) ผลเดิมไม่ถูกเขียนทับ

---

## ค่าจากเอกสาร (paper + Excel `[Gie]-Machine-Unlearning-USAI-CCA-2024.xlsx`)

| รายการ | ค่า | ที่มา |
|---|---|---|
| โมเดล unlearned ที่ใช้ใน paper | unlearn R2 = **unfreezeB4-B7** → downstream R2 = Block5a_se_excite-Block7 | Table 3/4, sheet `Result_unbalanced` |
| ผลเดิม (lab / field acc.) | original 0.84 / 0.65, unlearned 0.89 / 0.85 | Table 3/4 |
| downstream script | `USAI_unlearn/train.py` (split คงที่: `Traindf_fold4_8_v1.csv` + `Valdf_fold3_v1.csv`) — **ไม่ใช่** `train-Kfold.py` | sheet `EffNet-USAI_unlearn`, `USAI_EffNet(newData)` |
| downstream epochs | R1 = 200, R2 = 200 | Excel |
| unlearn R1 (FC) | 150+10+9+13+18 = **200 epochs** (resume 4 ครั้ง) | sheet `EffNet-ImageNet_unlearn` |
| unlearn R2 unfreezeB4-B7 | เทรน 115/200 แต่ใช้ **checkpoint epoch 50** (mini-ImageNet acc 0.91), lr **1e-6** | sheet `EffNet-ImageNet_unlearn` (Model Epoch) |
| GPU-hours | ดูจาก TensorBoard log ในตาราง Excel (เครื่อง 28) | — |

### ⚠️ จุดที่ paper กับโค้ดไม่ตรงกัน (ต้องยืนยันก่อนรัน / ก่อนแก้ paper)

1. **split mini-ImageNet** — paper/supplementary: 90/10 = 54,000 / 6,000 pairs; CSV ในโค้ด: train 42,000 / val 12,000 / test 6,000 (70/20/10)
2. **input resolution** — supplementary: 300×300; โค้ด EffNet-B5: 456×456
3. **TRUE pair** — supplementary บอก "both images drawn from the same class"; โค้ด (`Flip_generator`) ใช้ **ภาพเดียวกัน** augment ต่างกัน
4. **batch size** — downstream `train.py` = 8, unlearn `trainmodel.py` = 16 (default ทั้งคู่; paper ไม่ได้ระบุ)
5. **seed** — paper เขียนว่า "controlled using a predefined random seed" แต่โค้ดเดิมไม่มีการตั้ง seed

### ⚠️ Bug ใน original baseline ของ `train.py` (แก้แล้ว)

`--set MLorigin_USAI` สร้าง EfficientNet-B5 จาก `tf.keras.applications` ซึ่งมี `Rescaling(1/255)` + `Normalization` อยู่ในโมเดล
แต่ `data_loader.py` ก็ `rescale=1/255` แล้ว → ภาพถูกหาร 255 สองครั้ง เกือบดำทั้งภาพ
ผลจริง: `Models_USAI/R2/...EffNet-base_Block5a_se_excite-R2.h5` ทาย Normal ทุกภาพ (acc 65.4% = 857/1312)

แก้: เพิ่ม `--effnet_impl efn` ให้ original / C1 ใช้ `efficientnet.tfkeras` ตัวเดียวกับโมเดล unlearned และโมเดล legacy ใน paper
(script `run_downstream.sh` ใส่ให้อัตโนมัติ; ไม่ใส่ flag = พฤติกรรมเดิม)

### โมเดลที่ให้ตัวเลขใน paper (ยืนยันจาก notebook evaluation เดิม)

| Table | โมเดล | path |
|---|---|---|
| unlearned lab 0.89/0.83/0.71/0.76, field 0.85/0.21/0.21/0.17 | unlearn B4-B7 → downstream R2 Block5a | `$PAPER_UNLEARNED_H5` |
| original lab 0.84/0.77/0.57/0.63, field 0.65/0.25/0.13/0.16 | legacy | `$PAPER_ORIGINAL_H5` |

ข้อ 3 สำคัญกับ C2: control ที่ยุติธรรมต้องสร้าง pair แบบเดียวกับที่ใช้จริง (โค้ด = ภาพเดียวกัน)
planner ยังขอให้ยืนยันว่า flip ทำหลัง augmentation อื่นทั้งหมด — **ยืนยันจากโค้ดแล้ว**: `Flip_generator` เรียก `tf.image.flip_left_right` บนภาพที่ augment เสร็จแล้ว (exact pixel flip)

---

## วิธีรันบนเครื่อง 29

### 0. ดึงโค้ด + ตั้งค่า (ครั้งเดียว)

```bash
cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026
git pull
nano experiments/config_m29.sh      # ใส่ค่าทุกช่องที่เป็น __SET_ME__
```

ตรวจทุกอย่างก่อน (ไม่เทรน): path ทุกตัว, ภาพใน CSV มีจริงไหม, conda env, GPU

```bash
bash experiments/check_setup.sh
```

ทุกบรรทัดควรเป็น `[OK]` — ถ้า mini-ImageNet ขึ้น MISS เพราะ path ภาพยังเป็นของเครื่อง 28 ให้ตั้ง `MINI_PATH_REPLACE="/media/HDD=/media/tohn/HDD2"`

script เช็คค่าและ path ทุกตัวก่อนเทรน ถ้ายังไม่ได้ใส่หรือ path ไม่มีจริง จะหยุดทันทีพร้อมบอกว่าขาดอะไร
log ทุก run อยู่ที่ `$SAVE_DIR/logs/`

### 1. เช็ค evaluate.py กับโมเดลเดิม (ควรได้ตัวเลขตรง Table 3/4)

```bash
bash experiments/run_eval.sh paper-unlearned 1 0     # คาด lab acc 0.886, field acc 0.849
bash experiments/run_eval.sh paper-original  1 0     # คาด lab acc 0.835, field acc 0.648
```

### 2. C2 pre-train (1 รอบ: R1 → R2 unfreezeB4-B7)

```bash
nohup bash experiments/run_C2_pretrain.sh 0 > c2_pretrain.out 2>&1 &
```

> เช็ค val accuracy ของ C2 — ถ้าใกล้ 100% เร็วมาก แปลว่าโจทย์ง่าย ควรรายงานใน paper

### 3. downstream (R1 → R2) — `run_downstream.sh <model> <seed> <gpu>`

| model | seeds ที่ต้องรัน | หมายเหตุ |
|---|---|---|
| `original` | 1, 2, 3 | ใช้ `--effnet_impl efn` (ใส่ให้อัตโนมัติ) |
| `unlearned` | 2, 3 | ต้องตั้ง `UNLEARN_B4B7_CKPT` |
| `C1` | 1, 2, 3 | ต้องตั้ง `E_C1` (ดูข้อ 5) |
| `C2` | 1, 2, 3 | ต้องรันข้อ 2 ให้เสร็จก่อน |

ตัวอย่าง: 2 GPU รันคู่กัน

```bash
nohup bash -c 'for s in 1 2 3; do bash experiments/run_downstream.sh original  $s 0; done' > original.out 2>&1 &
nohup bash -c 'for s in 2 3;   do bash experiments/run_downstream.sh unlearned $s 1; done' > unlearned.out 2>&1 &
```

ถ้า R1 เสร็จแล้วแต่ R2 ค้าง สั่งคำสั่งเดิมซ้ำได้ (R1 ที่เสร็จแล้วจะถูกข้าม)

### 4. evaluate ทุกโมเดล (lab + field) — `run_eval.sh <model> <seed> <gpu>`

```bash
for m in original C1 C2; do for s in 1 2 3; do bash experiments/run_eval.sh $m $s 0; done; done
for s in 2 3; do bash experiments/run_eval.sh unlearned $s 0; done
```

ผลรวมอยู่ที่ `$RESULTS_CSV` และ prediction รายภาพที่ `$PRED_DIR/` (ใช้ทำ McNemar / bootstrap CI ของ A2 ได้)

### 5. หา `E_C1` สำหรับ C1

```bash
python3 experiments/compute_budget.py --unlearn_epochs 200 50 --n_usai_train <จำนวนแถวใน Traindf_fold4_8_v1.csv> \
    --tb_unlearn $TB_UNLEARN_R1 $TB_UNLEARN_R2 \
    --tb_downstream <Mylogs_tensor ของ original R2 seed 1 ที่เพิ่งเทรน>
```

ใส่ค่า "C1 --epochs (R2)" ลง `E_C1` ใน `config_m29.sh` (แนะนำค่าจากแบบ [B] GPU time)
หมายเหตุ: unlearn log อยู่เครื่อง 28 (3090 Ti) แต่ downstream เทรนบนเครื่อง 29 (2080 Ti) — ถ้าเทียบเวลา ต้องใช้ log จาก GPU รุ่นเดียวกัน หรือใช้แบบ [A]

### ผลเซฟที่ไหน (`$SAVE_DIR/EffNetB5Model/...`)

| model | R2 model folder |
|---|---|
| original | `MLorigin_USAI/weight_imagenet/R2_unbalanced/unfreezeBlock5a_se_excite/seedN/models/` |
| unlearned | `MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/seedN/models/` |
| C1 | `MLorigin_USAI_C1_computematched/weight_imagenet/R2_unbalanced/unfreezeBlock5a_se_excite/seedN/models/` |
| C2 | `MLunlearn_USAI_C2_sameimg/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/seedN/models/` |

---

## [5] สรุปผล

```bash
python3 experiments/seed_stats.py --csv $SAVE_DIR/results_seeds.csv --ref unlearned
```

ได้ mean ± SD ทุกโมเดล และ paired t-test (unlearned vs อื่นๆ) จับคู่ตาม seed

> n = 3 seeds → df = 2 ทดสอบมีกำลังต่ำ ควรรายงาน per-seed difference ด้วย (ทุก seed ไปทางเดียวกันหรือไม่)
