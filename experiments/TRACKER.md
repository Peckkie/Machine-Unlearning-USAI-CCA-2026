# Experiment Tracker — X1 / X2 (EfficientNet-B5)

> อัปเดตล่าสุด: **2026-10-02 10:45** · เครื่อง 29 (`yupaporn@10.177.191.29`, RTX 2080 Ti ×2, env `AI` tf 2.6.2)
>
> ดูสถานะจริงจาก log (อัตโนมัติ):
> ```bash
> ssh yupaporn@10.177.191.29 'cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026 && git pull -q && bash experiments/status.sh'
> ```

สถานะ: ⬜ ยังไม่เริ่ม · 🟡 กำลังรัน · ✅ เสร็จ · ❌ ล้มเหลว · ⏸️ รอตัดสินใจ

---

## 1. ตาราง run ทั้งหมด (12 รอบใหม่ + seed 1 เดิม)

### C2 pre-train (mini-ImageNet, same-image vs different-image, ไม่มี flip)

| ID | stage | ค่า | GPU | screen | สถานะ | เริ่ม | จบ | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|
| P1 | R1 transfer (FC) | 200 ep, lr 1e-5, bs 16 | 1 | `c2_pretrain` | ❌ → ⏸️ | 10-01 16:42 | 10-01 16:43 | OOM ที่ bs 16 บน 2080 Ti — รอเลือก เครื่อง 28 / bs 8 |
| P2 | R2 unfreeze B4-B7 | 50 ep, lr 1e-6, bs 16 | | | ⬜ | | | ต่อจาก P1 |

### Downstream (USAI 15AB: R1 FC 200 ep → R2 Block5a-B7 200 ep, bs 8)

| ID | model | seed | GPU | screen | R1 | R2 | Lab acc | Field acc | เริ่ม | จบ | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U1 | unlearned | 1 | — | — | ✅ | ✅ | **0.886** | **0.849** | — | — | โมเดลใน paper (ตรง Table 3/4) |
| O1 | original | 1 | 0 | `original` | ✅ val 0.7195 | 🟡 ep 69/200 val 0.878 | | | 10-01 16:43 | | ~4 นาที/epoch |
| O2 | original | 2 | 0 | `original` | ⬜ | ⬜ | | | | | ต่อจาก O1 อัตโนมัติ |
| O3 | original | 3 | 0 | `original` | ⬜ | ⬜ | | | | | ต่อจาก O2 อัตโนมัติ |
| U2 | unlearned | 2 | 1 | `unlearned` | 🟡 | ⬜ | | | 10-02 10:49 | | |
| U3 | unlearned | 3 | 1 | `unlearned` | ⬜ | ⬜ | | | | | ต่อจาก U2 อัตโนมัติ |
| C1-1 | C1 compute-matched | 1 | | | ⬜ | ⬜ | | | | | ⏸️ รอ `E_C1` |
| C1-2 | C1 compute-matched | 2 | | | ⬜ | ⬜ | | | | | ⏸️ |
| C1-3 | C1 compute-matched | 3 | | | ⬜ | ⬜ | | | | | ⏸️ |
| C2-1 | C2 same/diff image | 1 | | | ⬜ | ⬜ | | | | | ต้องรอ P2 |
| C2-2 | C2 same/diff image | 2 | | | ⬜ | ⬜ | | | | | ต้องรอ P2 |
| C2-3 | C2 same/diff image | 3 | | | ⬜ | ⬜ | | | | | ต้องรอ P2 |

### Smoke test (ไม่นับเป็นผล)

| ID | งาน | GPU | screen | สถานะ | ผล |
|---|---|---|---|---|---|
| S1 | original seed 99: R1 2 ep → R2 3 ep — เช็คว่าแก้ bug backbone สำเร็จ (R2 val_acc ต้อง > 0.65) | 1 | `smoke` | ✅ | R1 0.654 → R2 0.692 ✓ แก้สำเร็จ |

---

## 2. เรื่องที่รอตัดสินใจ

| # | เรื่อง | ทางเลือก | ตัดสินใจ |
|---|---|---|---|
| D1 | C2 pre-train OOM (bs 16 ไม่พอ 11 GB) | (1) รันบนเครื่อง 28 / 3090 Ti 24 GB — เหมือนต้นฉบับ (แนะนำ) · (2) bs 8 บนเครื่อง 29 · (3) bs 8 + gradient accumulation | |
| D2 | วิธี compute-match ของ C1 | (1) GPU time (แนะนำ) · (2) cap epoch เช่น 600 + กราฟ val_acc · (3) sample passes = 2,483 ep (~8–9 วัน/seed) | |
| D3 | ResNet-152V2 | ทำถ้า GPU ว่างหลังจบ EfficientNet-B5 | |

---

## 3. คำสั่งที่ใช้บ่อย

```bash
# สถานะทั้งหมด
ssh yupaporn@10.177.191.29 'cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026 && bash experiments/status.sh'
# เข้าไปดูหน้าจอ (ออก: Ctrl-A แล้ว D — ห้าม Ctrl-C)
ssh -t yupaporn@10.177.191.29 'screen -r original'
# เริ่ม run ใหม่ใน screen  (MODEL = original | unlearned | C1 | C2)
ssh yupaporn@10.177.191.29 'cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026 && screen -dmS NAME bash -c "bash experiments/run_downstream.sh MODEL SEED GPU; exec bash"'
# evaluate หลัง R2 เสร็จ
ssh yupaporn@10.177.191.29 'cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026 && bash experiments/run_eval.sh MODEL SEED GPU'
# สรุป mean ± SD + paired test
ssh yupaporn@10.177.191.29 'cd ~/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026 && /home/kannika/miniconda3/envs/AI/bin/python experiments/seed_stats.py --csv /media/tohn/HDD2/Model_unlearn_2026/results_seeds.csv --ref unlearned'
```

---

## 4. บันทึกเหตุการณ์ / ปัญหาที่เจอ

| วันที่ | เรื่อง | ผล / การแก้ |
|---|---|---|
| 10-01 | original baseline ใน `train.py` ใช้ `tf.keras.applications.EfficientNetB5` → rescale 1/255 ซ้ำ 2 ครั้ง โมเดลเดิมทาย Normal ทุกภาพ (acc 65.4%) | เพิ่ม `--effnet_impl efn` (ใช้ `efficientnet.tfkeras` เหมือน unlearned) — รอยืนยันด้วย S1 |
| 10-01 | original ใน paper เป็นโมเดล legacy (`ModelTrainByImages/R2_1/...`) คนละ pipeline | รัน original ใหม่ครบ 3 seeds (O1–O3) |
| 10-01 | env base (tf 2.3.1) ไม่เห็น GPU และโหลดโมเดล tf 2.6 ไม่ได้ (`keepdims`) | ใช้ env `AI` ของ kannika (tf 2.6.2, GPU 2 ตัว) |
| 10-01 | env `AI` activate script ใช้ `$LD_LIBRARY_PATH` → พังภายใต้ `set -u` | แก้ `activate_env` ใน `_common.sh` |
| 10-01 | ตรวจ `evaluate.py` กับโมเดลใน paper: lab 0.8864 / 0.826 / 0.710 / 0.755, field 0.8488 / 0.212 / 0.208 / 0.174 | ✅ ตรง Table 3/4 → วิธี evaluate ถูกต้อง |
| 10-01 | C2 pre-train OOM ที่ bs 16 (2080 Ti 11 GB) — ต้นฉบับเทรนบน 3090 Ti 24 GB | ⏸️ D1 |
| 10-02 | smoke test R2 val_acc 0.692 > 0.654; original seed 1 R2 ep 69 val 0.878 | ✅ ยืนยันแก้ bug backbone สำเร็จ |
| 10-01 | downstream ใช้เวลา ~4–5 นาที/epoch, GPU util ~15% (คอขวดที่ data loading) | 1 seed ≈ 1.5 วัน |

---

## 5. ค่าที่ใช้ (อ้างอิง)

| | ค่า | ที่มา |
|---|---|---|
| Train / Val | `Traindf_fold4_8_v1.csv` 4,601 / `Valdf_fold3_v1.csv` 656 | `train.py` |
| Lab / Field test | 1,312 / 807 ภาพ | `Testdf_fold1_2_v1.csv` / `UICCA_..._813Image.csv` |
| Unlearn R1 / R2 | 200 ep lr 1e-5 / checkpoint ep 50 lr 1e-6, bs 16 | Excel `EffNet-ImageNet_unlearn` |
| Downstream R1 / R2 | 200 ep lr 2e-5 / 200 ep lr 1e-5 (Block5a-B7), bs 8, RMSprop | Excel + `train.py` |
| ผลเก็บที่ | `/media/tohn/HDD2/Model_unlearn_2026/` (`logs/`, `results_seeds.csv`, `predictions/`) | `config_m29.sh` |
