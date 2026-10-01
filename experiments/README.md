# X1 / X2 — การทดลองเพิ่มเติมตอบกรรมการ (EfficientNet-B5)

| งาน | ตอบกรรมการ | รันใหม่ |
|---|---|---|
| X1-C1 ordinary fine-tuning (compute-matched) | ข้อ 2, 11 | downstream 3 seeds |
| X1-C2 unrelated auxiliary (same-image vs different-image) | ข้อ 2, 11 | pre-train 1 ครั้ง + downstream 3 seeds |
| X1-C3 pre-train แบบไม่มี flip augmentation | ข้อ 2, 11 | ไม่รัน → เขียนใน Limitations |
| X2 repeated runs 3 seeds (original, unlearned, C1, C2) | ข้อ 5 | original 2 + unlearned 2 seeds |

**รวม EfficientNet-B5 = 11 รอบใหม่** = C2 pre-train 1 + C1 ×3 + C2 ×3 + original ×2 + unlearned ×2
(ผล original / unlearned เดิมนับเป็น seed 1 → รันเพิ่ม `--seed 2` และ `--seed 3`)

> ⚠️ ทุก run ใช้ split เดียวกับผลใน paper (`train.py`) และ test set ชุดเดิม: Lab Testset1312 + Field Testset807

---

## สิ่งที่เพิ่มในโค้ด

| ไฟล์ | เพิ่ม |
|---|---|
| `CNNs_unlearn/data_generator.py` | `SameDiff_generator` — pair เดิม label เดิม (`cls`) augmentation เดิม แต่ label 0 = **ภาพอื่น** แทนภาพ flip (ไม่มี flip เลย) |
| `CNNs_unlearn/trainmodel.py` | `--task [flip, sameimg]` → `sameimg` เซฟใน `baseML_unlearn_sameimg/` |
| `USAI_unlearn/train.py`, `train-Kfold.py` | `--seed` (ตั้ง seed python/numpy/tf + data generator) และ `--tag` (เซฟใน `{set}_{tag}/`) |
| `experiments/config_m29.sh` | path / ค่าทั้งหมดของเครื่อง 29 (แก้ไฟล์นี้ไฟล์เดียว) |
| `experiments/run_C2_pretrain.sh`, `run_downstream.sh` | script รันพร้อมเช็ค path + log |
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
| unlearn R2 unfreezeB4-B7 | **115/200** (ไม่มีแถว resume) ⚠️ ต้องเช็คว่าเทรนจบที่ 115 จริงไหม | sheet `EffNet-ImageNet_unlearn` |
| GPU-hours | ดูจาก TensorBoard log ในตาราง Excel (เครื่อง 28) | — |

### ⚠️ จุดที่ paper กับโค้ดไม่ตรงกัน (ต้องยืนยันก่อนรัน / ก่อนแก้ paper)

1. **split mini-ImageNet** — paper/supplementary: 90/10 = 54,000 / 6,000 pairs; CSV ในโค้ด: train 42,000 / val 12,000 / test 6,000 (70/20/10)
2. **input resolution** — supplementary: 300×300; โค้ด EffNet-B5: 456×456
3. **TRUE pair** — supplementary บอก "both images drawn from the same class"; โค้ด (`Flip_generator`) ใช้ **ภาพเดียวกัน** augment ต่างกัน
4. **batch size** — `train.py` default 8, สคริปต์ baseline เก่า 16/32 → ต้องใช้ค่าเดียวกับที่ได้ผลใน paper
5. **seed** — paper เขียนว่า "controlled using a predefined random seed" แต่โค้ดเดิมไม่มีการตั้ง seed

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

script เช็คค่าและ path ทุกตัวก่อนเทรน ถ้ายังไม่ได้ใส่หรือ path ไม่มีจริง จะหยุดทันทีพร้อมบอกว่าขาดอะไร
log ทุก run อยู่ที่ `$SAVE_DIR/logs/`

### 1. C2 pre-train (1 รอบ: R1 → R2 unfreezeB4-B7)

```bash
nohup bash experiments/run_C2_pretrain.sh 0 > c2_pretrain.out 2>&1 &
```

> เช็ค val accuracy ของ C2 — ถ้าใกล้ 100% เร็วมาก แปลว่าโจทย์ง่าย ควรรายงานใน paper

### 2. downstream (R1 → R2) — `run_downstream.sh <model> <seed> <gpu>`

| model | seeds ที่ต้องรัน | หมายเหตุ |
|---|---|---|
| `original` | 2, 3 | seed 1 = ผลเดิม (ถ้า config เดิมตรงกัน) |
| `unlearned` | 2, 3 | ต้องตั้ง `UNLEARN_B4B7_CKPT` |
| `C1` | 1, 2, 3 | ต้องตั้ง `E_C1` (ดูข้อ 3) |
| `C2` | 1, 2, 3 | ต้องรันข้อ 1 ให้เสร็จก่อน |

ตัวอย่าง: 2 GPU รันคู่กัน

```bash
nohup bash -c 'for s in 2 3; do bash experiments/run_downstream.sh original  $s 0; done' > original.out 2>&1 &
nohup bash -c 'for s in 2 3; do bash experiments/run_downstream.sh unlearned $s 1; done' > unlearned.out 2>&1 &
```

ถ้า R1 เสร็จแล้วแต่ R2 ค้าง สั่งคำสั่งเดิมซ้ำได้ (R1 ที่เสร็จแล้วจะถูกข้าม)

### 3. หา `E_C1` สำหรับ C1

```bash
python3 experiments/compute_budget.py --unlearn_epochs 200 115 --n_usai_train <จำนวนแถวใน Traindf_fold4_8_v1.csv> \
    --tb_unlearn <Mylogs_tensor ของ unlearn R1> <Mylogs_tensor ของ unlearn R2 unfreezeB4-B7> \
    --tb_downstream <Mylogs_tensor ของ original R2>
```

ใส่ค่า "C1 --epochs (R2)" ลง `E_C1` ใน `config_m29.sh` (แนะนำค่าจากแบบ [B] GPU time)

### ผลเซฟที่ไหน (`$SAVE_DIR/EffNetB5Model/...`)

| model | R2 model folder |
|---|---|
| original | `MLorigin_USAI/weight_imagenet/R2_unbalanced/unfreezeBlock5a_se_excite/seedN/models/` |
| unlearned | `MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/seedN/models/` |
| C1 | `MLorigin_USAI_C1_computematched/weight_imagenet/R2_unbalanced/unfreezeBlock5a_se_excite/seedN/models/` |
| C2 | `MLunlearn_USAI_C2_sameimg/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/seedN/models/` |

---

## [5] สรุปผล

1. Evaluate ทุกโมเดล (notebook evaluation เดิม) บน lab test และ field test
2. กรอกผลลง `experiments/results_template.csv` (original/unlearned seed 1 = ผลเดิม)
3. รัน

```bash
python3 experiments/seed_stats.py --csv experiments/results_template.csv --ref unlearned
```

ได้ mean ± SD ทุกโมเดล และ paired t-test (unlearned vs อื่นๆ) จับคู่ตาม seed

> n = 3 seeds → df = 2 ทดสอบมีกำลังต่ำ ควรรายงาน per-seed difference ด้วย (ทุก seed ไปทางเดียวกันหรือไม่)
