#!/usr/bin/env bash
# =====================================================================================
#  X1 / X2 experiments — path config for machine 29 (yupaporn@10.177.191.29)
#  แก้เฉพาะไฟล์นี้ไฟล์เดียว ค่าที่เป็น __SET_ME__ ต้องใส่ก่อนรัน (script จะไม่ยอมรันถ้ายังไม่ได้ใส่)
# =====================================================================================

## ---------- code / environment ----------
REPO_DIR=$HOME/codes/USAI2026/Machine-Unlearning-USAI-CCA-2026
CONDA_ENV=/home/kannika/miniconda3/envs/AI   # probe_envs.sh: READY tf 2.6.2, เห็น GPU 2 ตัว — env เดียวกับที่เทรนโมเดลใน paper
                                              # สำรอง: bitnetenv2 (tf 2.6.0-rc0, GPU 2) ; base/conex ไม่เห็น GPU

## ---------- output (โฟลเดอร์ใหม่ ไม่ทับผลเดิม) ----------
SAVE_DIR=/media/tohn/HDD2/Model_unlearn_2026   # โฟลเดอร์ใหม่ (เปลี่ยนได้)

## ---------- datasets ----------
# folder ที่มี Traindf_fold4_8_v1.csv และ Valdf_fold3_v1.csv (split เดียวกับผลใน paper)
USAI_DATA_DIR=/media/tohn/HDD/VISION_dataset
# path ใน Excel เป็นของเครื่อง 28 (/media/HDD/...) — บนเครื่อง 29 อยู่ที่ /media/tohn/HDD2/...
# mini-ImageNet pair CSV (columns: img_path, cls, subset) — img_path ต้องชี้ไปที่ภาพบนเครื่อง 29
MINI_CSV=/home/kannika/code/mini-ImageNet_MachineUnlearn.csv   # CSV ที่แปลง path ภาพสำหรับเครื่อง 29 แล้ว (Code_Change_path_dataset_miniImageNet_..._for_Train29.ipynb)
MINI_PATH_REPLACE=""                      # ถ้า img_path ใน CSV ยังเป็นของเครื่อง 28 ใส่ "/media/HDD=/media/tohn/HDD2" (check_setup.sh จะบอก)

## ---------- test sets (ชุดเดียวกับ Table 3/4) ----------
LAB_TEST_CSV=/media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv                      # Lab Testset1312
FIELD_TEST_CSV=/home/yupaporn/CSV_file/UICCA_DiagRadioExp_AzureDb51Case813Image.csv      # Field Testset807 (Sub_Class_15AB != None)

## ---------- existing models (ผลเดิมใน paper) ----------
# โมเดลที่ให้ตัวเลขใน Table 3/4 (ยืนยันจาก notebook evaluation เดิม)
PAPER_UNLEARNED_H5=/media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/R2_unbalanced/unfreezeBlock5a_se_excite/exp_unfreezeB4-B7/models/modelEffNetB5_MLunlearn_USAI_unfreezeBlock5a_se_excite_exp_unfreezeB4-B7-R2_unbalanced.h5
PAPER_ORIGINAL_H5=/media/tohn/SSD/ModelTrainByImages/R2_1/models/B5R2_block5_15AB_1FC_3.h5   # legacy pipeline (ไม่ใช่ train.py)
# โมเดล unlearn บน mini-ImageNet: R2 unfreeze block4-block7 -> จุดเริ่มของ "unlearned" seed 2, 3
# Excel: ใช้ checkpoint epoch 50 (mini-ImageNet acc 0.91) ซึ่งอยู่เครื่อง 28:
#   /media/HDD/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/on_epoch_end/modelEffNetB5_Unlearning_unfreezeB4-B7_R2_epoch50.h5
#   -> เครื่อง 29 (/media/HDD -> /media/tohn/HDD2):
UNLEARN_B4B7_CKPT=/media/tohn/HDD2/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/on_epoch_end/modelEffNetB5_Unlearning_unfreezeB4-B7_R2_epoch50.h5
# TensorBoard log ของ unlearn (ใช้กับ compute_budget.py)
TB_UNLEARN_R1=/media/tohn/HDD2/mini-ImageNet/EffNetB5Model_unlearn/R1/Mylogs_tensor
TB_UNLEARN_R2=/media/tohn/HDD2/mini-ImageNet/EffNetB5Model_unlearn/R2/unfreezeB4-B7/Mylogs_tensor

## ---------- training settings (ต้องเท่ากับผลเดิม) ----------
BATCH_SIZE=8                              # downstream: train.py default (README เดิมไม่เคยใส่ --batchsize)
E_DS_R1=200                               # downstream R1 (FC) epochs
E_DS_R2=200                               # downstream R2 (Block5a_se_excite-Block7) epochs
LR_DS_R1=2e-5
LR_DS_R2=1e-5
DS_R2_NAME=unfreezeBlock5a_se_excite      # downstream R2 = Block5a_se_excite-Block7 (ตรงกับโมเดลใน paper)
BATCH_SIZE_UN=16                          # unlearn stage: trainmodel.py default
E_UN_R1=200                               # unlearn R1 epochs (Excel: 150+10+9+13+18)
E_UN_R2=50                                # unlearn R2 unfreezeB4-B7: เทรน 115 แต่ใช้ checkpoint epoch 50 (Excel 'Model Epoch')
LR_UN_R1=1e-5                             # Excel
LR_UN_R2=1e-6                             # Excel (ไม่ใช่ default 1e-5 ของ trainmodel.py)
E_C1=__SET_ME__                           # C1 downstream R2 epochs = E_DS_R2 + extra (จาก compute_budget.py)

## ---------- logs / results ----------
LOG_DIR=${SAVE_DIR}/logs
RESULTS_CSV=${SAVE_DIR}/results_seeds.csv     # input ของ seed_stats.py
PRED_DIR=${SAVE_DIR}/predictions              # prediction รายภาพ (McNemar / bootstrap CI)
