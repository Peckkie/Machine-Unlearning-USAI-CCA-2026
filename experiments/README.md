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

## ตั้งค่าก่อนรัน (เครื่อง 29)

```bash
cd Machine-Unlearning-USAI-CCA-2026
conda activate AI
SAVE=/media/tohn/HDD2/Model_unlearn
DATA=/media/tohn/HDD/VISION_dataset
MINI_CSV=/home/kannika/code/mini-ImageNet_MachineUnlearn.csv
E_UN_R1=200        # unlearn R1 (Excel)
E_UN_R2=115        # unlearn R2 unfreezeB4-B7 (Excel: 115/200 — ยืนยันก่อน)
BS=8               # batch size downstream — ยืนยันก่อน
```

---

## [1] C2 — pre-train same-image vs different-image (1 รอบ)

ใช้ค่าทุกอย่างเหมือน unlearn เดิม เปลี่ยนแค่ `--task sameimg`

```bash
cd CNNs_unlearn
# R1 transfer
python3 trainmodel.py --gpu 0 --network_name EffNetB5 --weight imagenet --set baseML_unlearn \
    --data_path $MINI_CSV --save_dir $SAVE --name transfer --R 1 --epochs $E_UN_R1 --task sameimg
# R2 unfreeze block4-block7 (variant เดียวกับโมเดล unlearned ใน paper)
python3 trainmodel.py --gpu 0 --network_name EffNetB5 --weight imagenet --set baseML_unlearn \
    --data_path $MINI_CSV --save_dir $SAVE --name unfreezeB4-B7 --R 2 --epochs $E_UN_R2 --task sameimg \
    --checkpoint_dir $SAVE/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R1/transfer/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_transfer-R1.h5
cd ..
```

> เช็ค val accuracy ของ C2 — ถ้าใกล้ 100% เร็วมาก แปลว่าโจทย์ง่าย ควรรายงานไว้ใน paper ด้วย

## [2] C2 — downstream 3 seeds

```bash
cd USAI_unlearn
C2_CKPT=$SAVE/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB4-B7/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB4-B7-R2.h5
for SEED in 1 2 3; do
  # R1 transfer
  python3 train.py --gpu 0 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --tag C2_sameimg \
      --data_path $DATA --save_dir $SAVE --data unbalanced --name transfer --R 1 --exp unfreezeB4-B7 \
      --checkpoint_dir $C2_CKPT --batchsize $BS --seed $SEED
  # R2 fine-tune Block5a_se_excite-Block7
  R1_DIR=$SAVE/EffNetB5Model/MLunlearn_USAI_C2_sameimg/R1_unbalanced/transfer_exp_unfreezeB4-B7/seed$SEED/models
  R1_NAME=modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeB4-B7-R1_unbalanced_C2_sameimg_seed$SEED
  python3 train.py --gpu 0 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --tag C2_sameimg \
      --data_path $DATA --save_dir $SAVE --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --exp unfreezeB4-B7 \
      --checkpoint_dir $R1_DIR/$R1_NAME.weights.h5 --Modeljson_dir $R1_DIR/$R1_NAME.json --batchsize $BS --seed $SEED
done
cd ..
```

## [3] C1 — ordinary fine-tuning, compute-matched (3 seeds)

**ขั้นที่ 1: หาจำนวน epoch** — ใช้ TensorBoard log ของ run เดิม (path อยู่ใน Excel)

```bash
python3 experiments/compute_budget.py --unlearn_epochs $E_UN_R1 $E_UN_R2 --n_usai_train <จำนวนแถวใน Traindf_fold4_8_v1.csv> \
    --tb_unlearn <Mylogs_tensor ของ unlearn R1> <Mylogs_tensor ของ unlearn R2 unfreezeB4-B7> \
    --tb_downstream <Mylogs_tensor ของ original R2>
E_C1=<ค่า "C1 --epochs (R2)" ที่ได้>
```

**ขั้นที่ 2: เทรน** — pipeline เดียวกับ original (ImageNet → R1 → R2) แต่ R2 ยาวขึ้นเป็น `$E_C1`

```bash
cd USAI_unlearn
for SEED in 1 2 3; do
  python3 train.py --gpu 1 --network_name EffNetB5 --weight imagenet --set MLorigin_USAI --tag C1_computematched \
      --data_path $DATA --save_dir $SAVE --data unbalanced --name transfer --R 1 --batchsize $BS --seed $SEED
  R1_DIR=$SAVE/EffNetB5Model/MLorigin_USAI_C1_computematched/weight_imagenet/R1_unbalanced/transfer/seed$SEED/models
  R1_NAME=modelEffNetB5_MLorigin_USAI_transfer-R1_unbalanced_C1_computematched_seed$SEED
  python3 train.py --gpu 1 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLorigin_USAI --tag C1_computematched \
      --data_path $DATA --save_dir $SAVE --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --epochs $E_C1 \
      --checkpoint_dir $R1_DIR/$R1_NAME.weights.h5 --Modeljson_dir $R1_DIR/$R1_NAME.json --batchsize $BS --seed $SEED
done
cd ..
```

> checkpoint ถูกเซฟทุก 20 epoch ใน `on_epoch_end/` → ใช้ดูได้ว่า performance ตันหรือ overfit ที่ epoch ไหน
> ถ้าเทรนค้าง ใช้ `--resume --epochendName on_epoch_end_resume` แบบเดิม (ใส่ `--tag` และ `--seed` เดิมด้วย)

## [4] X2 — original และ unlearned seed 2, 3

ใช้ `train.py` แบบ [3] แต่ **ไม่ใส่ `--tag`** และ R2 ใช้ `--epochs 200`
- original: `--set MLorigin_USAI`
- unlearned: `--set MLunlearn_USAI --exp unfreezeB4-B7 --checkpoint_dir <โมเดล unlearn R2 unfreezeB4-B7 เดิม>`

เพิ่ม `--seed 2` / `--seed 3` → ผลเซฟใน sub-folder `seed2/`, `seed3/` ไม่ทับผลเดิม

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
