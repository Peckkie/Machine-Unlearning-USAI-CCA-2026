#!/usr/bin/env bash
# =====================================================================================
#  X1 / X2 experiments — path config for machine 29 (yupaporn@10.177.191.29)
#  แก้เฉพาะไฟล์นี้ไฟล์เดียว ค่าที่เป็น __SET_ME__ ต้องใส่ก่อนรัน (script จะไม่ยอมรันถ้ายังไม่ได้ใส่)
# =====================================================================================

## ---------- code / environment ----------
REPO_DIR=$HOME/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026
CONDA_ENV=AI

## ---------- output (โฟลเดอร์ใหม่ ไม่ทับผลเดิม) ----------
SAVE_DIR=__SET_ME__                       # e.g. /media/tohn/HDD2/Model_unlearn_2026

## ---------- datasets ----------
# folder ที่มี Traindf_fold4_8_v1.csv และ Valdf_fold3_v1.csv (split เดียวกับผลใน paper)
USAI_DATA_DIR=/media/tohn/HDD/VISION_dataset
# mini-ImageNet pair CSV (columns: img_path, cls, subset) — img_path ต้องชี้ไปที่ภาพบนเครื่อง 29
MINI_CSV=__SET_ME__                       # e.g. /home/kannika/code/mini-ImageNet_MachineUnlearn.csv

## ---------- existing models (ผลเดิมใน paper) ----------
# โมเดล unlearn บน mini-ImageNet: R2 unfreeze block4-block7 (.h5)  -> ใช้เป็นจุดเริ่มของ "unlearned" seed 2, 3
UNLEARN_B4B7_CKPT=__SET_ME__              # e.g. /media/tohn/HDD2/Model_unlearn/ModelsR2_MiniImageNet/modelEffNetB5_Unlearning_unfreezeB4-B7_R2.h5

## ---------- training settings (ต้องเท่ากับผลเดิม) ----------
BATCH_SIZE=__SET_ME__                     # batch size downstream ที่ใช้ได้ผลใน paper (train.py default = 8)
E_DS_R1=200                               # downstream R1 (FC) epochs
E_DS_R2=200                               # downstream R2 (Block5a_se_excite-Block7) epochs
LR_DS_R1=2e-5
LR_DS_R2=1e-5
E_UN_R1=200                               # unlearn R1 epochs (Excel: 150+10+9+13+18)
E_UN_R2=__SET_ME__                        # unlearn R2 unfreezeB4-B7 epochs (Excel: 115/200 — ยืนยัน)
E_C1=__SET_ME__                           # C1 downstream R2 epochs = E_DS_R2 + extra (จาก compute_budget.py)

## ---------- logs ----------
LOG_DIR=${SAVE_DIR}/logs
