#!/usr/bin/env bash
# =====================================================================================
#  X1 / X2 experiments — path config for machine 28 (RTX 3090 Ti, 24 GB)
#  เปิดใช้บนเครื่อง 28 ครั้งเดียว:   echo 28 > experiments/.machine
#  แล้วตรวจ:                       bash experiments/check_setup.sh   (+ probe_envs.sh เพื่อหา conda env)
#  path ด้านล่างมาจาก Excel / README / notebook เดิมของเครื่อง 28 — ค่าที่มี (ยืนยัน) ต้องเช็คด้วย check_setup.sh
# =====================================================================================

## ---------- code / environment ----------
REPO_DIR=$HOME/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026
CONDA_ENV=/home/yupaporn/miniconda3/envs/unlearn26   # สร้าง 10-02 (~/setup_unlearn26.sh): tf 2.6.2 + efficientnet 1.0.0 = version เดียวกับ env AI ของเครื่อง 29

## ---------- output ----------
SAVE_DIR=/media/HDD/Model_unlearn_2026    # (ยืนยัน) โฟลเดอร์ใหม่บน HDD ของเครื่อง 28

## ---------- datasets ----------
# USAI: README เดิมของเครื่อง 28 ใช้ /media/HDD/VISION_dataset/CSV  (ต้องมี Traindf_fold4_8_v1.csv + Valdf_fold3_v1.csv)
USAI_DATA_DIR=/media/HDD/Model_unlearn_2026/usai_csv   # copy จากเครื่อง 29 (10-02, md5 ตรง): Traindf_fold4_8_v1.csv 4,601 + Valdf_fold3_v1.csv 656 — ภาพมีครบหลังแปลง path
# "Path Crop" ใน CSV เป็น path ของเครื่อง 29 (/media/tohn/HDD/VISION_dataset/USAI/...) -> แปลงเป็นของเครื่อง 28
USAI_PATH_REPLACE="/media/tohn/HDD=/media/HDD"   # ยืนยันแล้ว 10-02: ภาพ train/val มีครบ 100%
# mini-ImageNet pair CSV เดิมของเครื่อง 28 (mini-ImageNet-Dataset.ipynb) — img_path = /media/HDD/mini-ImageNet/mini-imagenet/...
MINI_CSV=/home/kannika/codes_AI/CSV/mini-ImageNet_MachineUnlearn.csv   # (ยืนยัน)
MINI_PATH_REPLACE=""

## ---------- test sets (ใช้ตอน evaluate; evaluate บนเครื่อง 29 ก็ได้) ----------
LAB_TEST_CSV=/media/HDD/VISION_dataset/CSV/Testdf_fold1_2_v1.csv         # (ยืนยัน)
FIELD_TEST_CSV=__SET_ME__                                                # ภาพ field test อยู่เครื่อง 29 (/media/tohn/SSD/...) -> evaluate บนเครื่อง 29

## ---------- existing models ----------
PAPER_UNLEARNED_H5=__SET_ME__             # อยู่เครื่อง 29 — evaluate บนเครื่อง 29
PAPER_ORIGINAL_H5=__SET_ME__
# โมเดล unlearn บน mini-ImageNet: R2 unfreeze B4-B7 checkpoint epoch 50 (Excel) — ต้นฉบับอยู่เครื่องนี้
UNLEARN_B4B7_CKPT=/media/HDD/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/on_epoch_end/modelEffNetB5_Unlearning_unfreezeB4-B7_R2_epoch50.h5
TB_UNLEARN_R1=/media/HDD/mini-ImageNet/EffNetB5Model_unlearn/R1/Mylogs_tensor
TB_UNLEARN_R2=/media/HDD/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/Mylogs_tensor

## ---------- training settings (เหมือนเครื่อง 29 / ต้นฉบับ) ----------
BATCH_SIZE=8
E_DS_R1=${E_DS_R1:-200}
E_DS_R2=${E_DS_R2:-200}
LR_DS_R1=2e-5
LR_DS_R2=1e-5
DS_R2_NAME=unfreezeBlock5a_se_excite
BATCH_SIZE_UN=16                          # 3090 Ti 24 GB รับ batch 16 ได้ = เท่าโมเดล unlearned ต้นฉบับ (เครื่อง 29 OOM)
E_UN_R1=200
E_UN_R2=50
LR_UN_R1=1e-5
LR_UN_R2=1e-6
E_C1=__SET_ME__

## ---------- logs / results ----------
LOG_DIR=${SAVE_DIR}/logs
RESULTS_CSV=${SAVE_DIR}/results_seeds.csv
PRED_DIR=${SAVE_DIR}/predictions
