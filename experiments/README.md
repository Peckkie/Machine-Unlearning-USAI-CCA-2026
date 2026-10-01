# X1 / X2 — การทดลองเพิ่มเติมตอบกรรมการ (EfficientNet-B5)

| งาน | ตอบกรรมการ | รันใหม่ |
|---|---|---|
| X1-C1 ordinary fine-tuning (compute-matched) | ข้อ 2, 11 | downstream 3 seeds |
| X1-C2 unrelated auxiliary (same-image vs different-image) | ข้อ 2, 11 | pre-train 1 ครั้ง + downstream 3 seeds |
| X1-C3 pre-train แบบไม่มี flip augmentation | ข้อ 2, 11 | ไม่รัน → เขียนใน Limitations |
| X2 repeated runs 3 seeds (original, unlearned, C1, C2) | ข้อ 5 | original 2 + unlearned 2 seeds |

**รวม EfficientNet-B5 = 11 รอบใหม่** = C2 pre-train 1 + C1 ×3 + C2 ×3 + original ×2 + unlearned ×2
(ผล original / unlearned เดิมนับเป็น seed 1 → รันเพิ่ม `--seed 2` และ `--seed 3`)

> ⚠️ ทุก run ต้องใช้ **fold เดียวกัน** กับผล original / unlearned เดิม และใช้ test set (lab + field) ชุดเดิม

---

## สิ่งที่เพิ่มในโค้ด

| ไฟล์ | เพิ่ม |
|---|---|
| `CNNs_unlearn/data_generator.py` | `SameDiff_generator` — pair เดิม label เดิม (`cls`) augmentation เดิม แต่ label 0 = **ภาพอื่น** แทนภาพ flip (ไม่มี flip เลย) |
| `CNNs_unlearn/trainmodel.py` | `--task [flip, sameimg]` → `sameimg` เซฟใน `baseML_unlearn_sameimg/` |
| `USAI_unlearn/train-Kfold.py` | `--seed` (ตั้ง seed python/numpy/tf + data generator) และ `--tag` (เซฟใน `{set}_{tag}/`) |
| `experiments/compute_budget.py` | คำนวณจำนวน epoch ที่ต้องเพิ่มให้ C1 |
| `experiments/seed_stats.py` | mean ± SD + paired t-test ข้าม seeds |

ไม่ใส่ `--seed` / `--tag` / `--task` = ทำงานเหมือนเดิมทุกอย่าง (path เดิม ชื่อไฟล์เดิม) ผลเดิมไม่ถูกเขียนทับ

---

## ตั้งค่าก่อนรัน (เครื่อง 29)

```bash
cd Machine-Unlearning-USAI-CCA-2026
conda activate AI
FOLD=4                                   # fold เดียวกับผลเดิม
SAVE=/media/tohn/HDD2/Model_unlearn
MINI_CSV=/home/kannika/code/mini-ImageNet_MachineUnlearn.csv
E_UN_R1=200                              # epoch ของ unlearn R1 เดิม (ต้องเท่าของเดิม)
E_UN_R2=200                              # epoch ของ unlearn R2 เดิม (ต้องเท่าของเดิม)
```

---

## [1] C2 — pre-train same-image vs different-image (1 รอบ)

ใช้ค่าทุกอย่างเหมือน unlearn เดิม เปลี่ยนแค่ `--task sameimg`

```bash
cd CNNs_unlearn
# R1 transfer
python3 trainmodel.py --gpu 0 --network_name EffNetB5 --weight imagenet --set baseML_unlearn \
    --data_path $MINI_CSV --save_dir $SAVE --name transfer --R 1 --epochs $E_UN_R1 --task sameimg
# R2 unfreeze (ใช้ variant เดียวกับโมเดล unlearned ที่ใช้ใน paper)
python3 trainmodel.py --gpu 0 --network_name EffNetB5 --weight imagenet --set baseML_unlearn \
    --data_path $MINI_CSV --save_dir $SAVE --name unfreezeB5a_se_excite --R 2 --epochs $E_UN_R2 --task sameimg \
    --checkpoint_dir $SAVE/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R1/transfer/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_transfer-R1.h5
cd ..
```

> เช็ค val accuracy ของ C2 — ถ้าใกล้ 100% เร็วมาก แปลว่าโจทย์ง่าย ควรรายงานไว้ใน paper ด้วย

## [2] C2 — downstream 3 seeds

```bash
cd USAI_unlearn
C2_CKPT=$SAVE/EffNetB5Model/baseML_unlearn_sameimg/weight_imagenet/R2/unfreezeB5a_se_excite/models/modelEffNetB5_Unlearning_miniImageNet_sameimg_unfreezeB5a_se_excite-R2.h5
for SEED in 1 2 3; do
  # R1 transfer
  python3 train-Kfold.py --gpu 0 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --tag C2_sameimg \
      --data unbalanced --name transfer --R 1 --exp unfreezeBlock5a_se_excite --checkpoint_dir $C2_CKPT --fold $FOLD --seed $SEED
  # R2 fine-tune
  R1_DIR=$SAVE/EffNetB5Model/MLunlearn_USAI_C2_sameimg/run_Kfold/R1_unbalanced/transfer_exp_unfreezeBlock5a_se_excite/fold$FOLD/seed$SEED/models
  R1_NAME=modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeBlock5a_se_excite-R1_unbalanced_fold${FOLD}_C2_sameimg_seed$SEED
  python3 train-Kfold.py --gpu 0 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --tag C2_sameimg \
      --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --exp unfreezeBlock5a_se_excite \
      --checkpoint_dir $R1_DIR/$R1_NAME.weights.h5 --Modeljson_dir $R1_DIR/$R1_NAME.json --fold $FOLD --seed $SEED
done
```

## [3] C1 — ordinary fine-tuning, compute-matched (3 seeds)

**ขั้นที่ 1: หาจำนวน epoch** — ใช้ TensorBoard log ของ run เดิม (วัดเวลา GPU จริง)

```bash
python3 ../experiments/compute_budget.py --unlearn_epochs $E_UN_R1 $E_UN_R2 --n_usai_train <จำนวนภาพ train ของ fold> \
    --tb_unlearn <Mylogs_tensor ของ unlearn R1> <Mylogs_tensor ของ unlearn R2> \
    --tb_downstream <Mylogs_tensor ของ original R2 เดิม>
E_C1=<ค่า "C1 --epochs (R2)" ที่ได้>
```

**ขั้นที่ 2: เทรน** — pipeline เดียวกับ original (ImageNet → R1 → R2) แต่ R2 ยาวขึ้นเป็น `$E_C1`

```bash
for SEED in 1 2 3; do
  python3 train-Kfold.py --gpu 1 --network_name EffNetB5 --weight imagenet --set MLorigin_USAI --tag C1_computematched \
      --data unbalanced --name transfer --R 1 --fold $FOLD --seed $SEED
  R1_DIR=$SAVE/EffNetB5Model/MLorigin_USAI_C1_computematched/weight_imagenet/R1_unbalanced/transfer/seed$SEED/models
  R1_NAME=modelEffNetB5_MLorigin_USAI_transfer-R1_unbalanced_C1_computematched_seed$SEED
  python3 train-Kfold.py --gpu 1 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLorigin_USAI --tag C1_computematched \
      --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --epochs $E_C1 \
      --checkpoint_dir $R1_DIR/$R1_NAME.weights.h5 --Modeljson_dir $R1_DIR/$R1_NAME.json --fold $FOLD --seed $SEED
done
```

> checkpoint ถูกเซฟทุก 20 epoch ใน `on_epoch_end/` → ใช้ดูได้ว่า performance ตันหรือ overfit ที่ epoch ไหน
> ถ้าเทรนค้าง ใช้ `--resume --epochendName on_epoch_end_resume` แบบเดิม (ใส่ `--tag` และ `--seed` เดิมด้วย)

## [4] X2 — original และ unlearned seed 2, 3

คำสั่งเดิมตาม README หลัก **ไม่ใส่ `--tag`** แต่เพิ่ม `--seed 2` / `--seed 3`
ผลจะเซฟใน sub-folder `seed2/`, `seed3/` ไม่ทับผลเดิม

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
