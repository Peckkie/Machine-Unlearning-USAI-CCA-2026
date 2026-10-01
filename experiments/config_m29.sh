#!/usr/bin/env bash
# =====================================================================================
#  X1 / X2 experiments — path config for machine 29 (yupaporn@10.177.191.29)
#  แก้เฉพาะไฟล์นี้ไฟล์เดียว ค่าที่เป็น __SET_ME__ ต้องใส่ก่อนรัน (script จะไม่ยอมรันถ้ายังไม่ได้ใส่)
# =====================================================================================

## ---------- code / environment ----------
REPO_DIR=$HOME/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026
CONDA_ENV=__SET_ME__                      # env ที่มี tensorflow + efficientnet + scikit-image (เช็ค: python -c "import efficientnet.tfkeras")
                                          # usai10k-python3.9 ใช้ evaluate ได้ แต่ยังไม่รู้ว่ามี efficientnet ไหม; env เดิมของโปรเจกต์คือ AI

## ---------- output (โฟลเดอร์ใหม่ ไม่ทับผลเดิม) ----------
SAVE_DIR=__SET_ME__                       # e.g. /media/tohn/HDD2/Model_unlearn_2026

## ---------- datasets ----------
# folder ที่มี Traindf_fold4_8_v1.csv และ Valdf_fold3_v1.csv (split เดียวกับผลใน paper)
USAI_DATA_DIR=/media/tohn/HDD/VISION_dataset
# mini-ImageNet pair CSV (columns: img_path, cls, subset) — img_path ต้องชี้ไปที่ภาพบนเครื่อง 29
MINI_CSV=__SET_ME__                       # e.g. /home/kannika/code/mini-ImageNet_MachineUnlearn.csv

## ---------- test sets (ชุดเดียวกับ Table 3/4) ----------
LAB_TEST_CSV=/media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv                      # Lab Testset1312
FIELD_TEST_CSV=/home/yupaporn/CSV_file/UICCA_DiagRadioExp_AzureDb51Case813Image.csv      # Field Testset807 (Sub_Class_15AB != None)

## ---------- existing models (ผลเดิมใน paper) ----------
# โมเดลที่ให้ตัวเลขใน Table 3/4 (ยืนยันจาก notebook evaluation เดิม)
PAPER_UNLEARNED_H5=/media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/models/modelEffNetB5_MLunlearn_USAI_unfreezeBlock5a_se_excite_exp_unfreezeB4-B7-R2_unbalanced.h5
PAPER_ORIGINAL_H5=/media/tohn/SSD/ModelTrainByImages/R2_1/models/B5R2_block5_15AB_1FC_3.h5   # legacy pipeline (ไม่ใช่ train.py)
# โมเดล unlearn บน mini-ImageNet: R2 unfreeze block4-block7 (.h5)  -> ใช้เป็นจุดเริ่มของ "unlearned" seed 2, 3
UNLEARN_B4B7_CKPT=__SET_ME__              # e.g. /media/tohn/HDD2/Model_unlearn/ModelsR2_MiniImageNet/modelEffNetB5_Unlearning_unfreezeB4-B7_R2.h5
#      (เครื่อง 28: /media/HDD/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/...)

## ---------- training settings (ต้องเท่ากับผลเดิม) ----------
BATCH_SIZE=__SET_ME__                     # batch size downstream (train.py default = 8) — GPU เครื่อง 29 = RTX 2080 Ti 11GB
E_DS_R1=200                               # downstream R1 (FC) epochs
E_DS_R2=200                               # downstream R2 (Block5a_se_excite-Block7) epochs
LR_DS_R1=2e-5
LR_DS_R2=1e-5
DS_R2_NAME=unfreezeBlock5a_se_excite      # downstream R2 = Block5a_se_excite-Block7 (ตรงกับโมเดลใน paper)
E_UN_R1=200                               # unlearn R1 epochs (Excel: 150+10+9+13+18)
E_UN_R2=__SET_ME__                        # unlearn R2 unfreezeB4-B7 epochs (Excel: 115/200 — ยืนยัน)
E_C1=__SET_ME__                           # C1 downstream R2 epochs = E_DS_R2 + extra (จาก compute_budget.py)

## ---------- logs / results ----------
LOG_DIR=${SAVE_DIR}/logs
RESULTS_CSV=${SAVE_DIR}/results_seeds.csv     # input ของ seed_stats.py
PRED_DIR=${SAVE_DIR}/predictions              # prediction รายภาพ (McNemar / bootstrap CI)
