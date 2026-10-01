# Machine Unlearning USAI CCA 2026

## 📂 Dataset - USAI15AB 8-fold Validation

## 🖥️ [1] เครื่อง 28

```
# Train - Validation Set
/media/HDD/VISION_dataset/CSV/Traindf_fold3_10_Kfold.csv

# Test Set
/media/HDD/VISION_dataset/CSV/Testdf_fold1_2_v1.csv
```

## 🖥️ [2] เครื่อง 29

```
# Train - Validation Set
/media/tohn/HDD/VISION_dataset/Traindf_fold3_10_Kfold.csv

# Test Set
/media/tohn/HDD/VISION_dataset/Testdf_fold1_2_v1.csv
```

-------------------------------------------------------------------------------------------------------------------------------------------------------

## Training ML Unlearn on USAI15AB Dataset  - 8 fold Validation

### 💡 [1] Train EffNetB5 Network

- เทรน ```EffNetB5``` ทั้งหมดสามารถเทรนบนเครื่อง 29 ได้เลย เพราะโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` อยู่ที่เครื่อง 29 ทั้งหมด
- ลักษณะการเทรนจะเป็นการนำโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` ด้วยแบบต่างๆ ```(unfreeze block4, unfreezeB4-B7, unfreezeB1-B4, unfreeze Block5a_se_excite-Block7)``` เอามา Fine-Tune เฉพาะ ```Block5a_se_excite-Block7``` บนชุดข้อมูล USAI15AB แบบ 8-fold Validation (เพราะ Block5a_se_excite-Block7 แม่นสุดในการเทรน EffNetB5 แบบธรรมดาไม่มีการ Unlearn)
  

- [x] **Transfer Learning**

```
# Train R1
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate AI
python3 train-Kfold.py --gpu 0 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --data unbalanced --name transfer --R 1 --exp unfreezeBlock5a_se_excite --checkpoint_dir /media/tohn/HDD2/Model_unlearn/ModelsR2_MiniImageNet/modelEffNetB5_Unlearning_unfreezeB5a_se_excite_R2.h5 --fold 4

# Resume R1 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 0 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --data unbalanced --name transfer --R 1 --exp unfreezeBlock5a_se_excite --checkpoint_dir /media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreezeBlock5a_se_excite/fold4/on_epoch_end/modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeBlock5a_se_excite-R1_unbalanced_fold4.h5 --fold 4 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreezeB4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/media/tohn/HDD2/Model_unlearn/ModelsR2_MiniImageNet``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10]
    
- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1 แบบ **resume**
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreezeB4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ```--checkpoint_dir``` เปลี่ยน ตาม path ในโฟลเดอร์ on_epoch_end ```/path/to/on_epoch_end/R1/foldk/modelname.h5``` ให้ related กับ ```--exp``` ที่เทรนค้างเอาไว้ , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ


- [x] **Fine-Tuning**
      
```
# Train R2
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate AI
python3 train-Kfold.py --gpu 0 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --exp unfreezeBlock5a_se_excite --checkpoint_dir /media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreezeBlock5a_se_excite/fold4/models/modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeBlock5a_se_excite-R1_unbalanced_fold4.weights.h5 --Modeljson_dir /media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreezeBlock5a_se_excite/fold4/models/modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeBlock5a_se_excite-R1_unbalanced_fold4.json --fold 4

# Resume R2 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 0 --lr 1e-5 --network_name EffNetB5 --weight imagenet --set MLunlearn_USAI --data unbalanced --name unfreezeBlock5a_se_excite --R 2 --exp unfreezeBlock5a_se_excite --checkpoint_dir /media/tohn/HDD2/Model_unlearn/EffNetB5Model/MLunlearn_USAI/run_Kfold/R2_unbalanced/transfer_exp_unfreezeBlock5a_se_excite/fold4/on_epoch_end/modelEffNetB5_MLunlearn_USAI_transfer_exp_unfreezeBlock5a_se_excite-R2_unbalanced_fold4.h5 --fold 4 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2
- [ ] ```--lr``` เปลี่ยนเป็น 1e-5, ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreezeB4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/R1/foldk/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--Modeljson_dir``` เปลี่ยน ตาม path ใน  ```path/to/model/R1/foldk/modelname.json``` ให้ related กับ ```--exp```,  ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9,10]

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2 แบบ **resume**
- [ ] ```--lr``` เปลี่ยนเป็น 1e-5, ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreezeB4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/R1/foldk/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--Modeljson_dir``` เปลี่ยน ตาม path ใน  ```path/to/model/R1/foldk/modelname.json``` ให้ related กับ ```--exp```,  ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9,10], เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ

-------------------------------------------------------------------------------------------------------------------------------------------------------

### 💡 [2] Train ResNet152v2 Network

- เทรน ```ResNet152v2``` ต้องเช็คก่อนว่าโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` อยู่ที่เครื่อง 28 หรือ 29 ตาม path ที่อยู่ใน Excel ชื่อ Sheet ```ResNet-ImageNet_unlearn```
- ลักษณะการเทรนจะเป็นการนำโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` ด้วยแบบต่างๆ ```[unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block]``` เอามา Fine-Tune เฉพาะ ```unfreeze_conv3_block-conv5_block``` บนชุดข้อมูล USAI15AB แบบ 8-fold Validation (เพราะ unfreeze_conv3_block-conv5_block แม่นสุดในการเทรน ResNet152v2 แบบธรรมดาไม่มีการ Unlearn)

### 💾 **[2.1] Train บนเครื่อง 28** 

- [x] **Transfer Learning**

```
# Train R1
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate tensorflow
python3 train-Kfold.py --gpu 1 --R 1 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name transfer --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/HDD/mini-ImageNet/ResNet152v2Model/baseML_unlearn/R2/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_Unlearning_miniImageNet-R2.json --checkpoint_dir /media/HDD/mini-ImageNet/ResNet152v2Model/baseML_unlearn/R2/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_Unlearning_miniImageNet-R2.weights.h5 \
        --weight imagenet --data_path /media/HDD/VISION_dataset/CSV --save_dir /media/HDD/mini-ImageNet \
        --imgsize 224 --fold 5

# Resume R1 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 1 --R 1 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name transfer --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --data_path /media/HDD/VISION_dataset/CSV --save_dir /media/HDD/mini-ImageNet \
        --imgsize 224 --fold 5 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม /path/to/model/modelname.json ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10]
    
- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1 แบบ **resume**
- [ ]   ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/on_epoch_end/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/on_epoch_end/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ


- [x] **Fine-Tuning**
      
```
# Train R2
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate tensorflow
python3 train-Kfold.py --gpu 1 --R 2 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name unfreeze_conv3_block-conv5_block --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/models/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --data_path /media/HDD/VISION_dataset/CSV --save_dir /media/HDD/mini-ImageNet \
        --imgsize 224 --fold 5 --lr 1e-5

# Resume R2 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 1 --R 2 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name unfreeze_conv3_block-conv5_block --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R2_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R2_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv3_block-conv5_block-R2_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --data_path /media/HDD/VISION_dataset/CSV --save_dir /media/HDD/mini-ImageNet \
        --imgsize 224 --fold 5 --lr 1e-5 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/model/R1/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/R1/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], ```--lr``` เปลี่ยนเป็น 1e-5
    
- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2 แบบ **resume**
- [ ]   ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/on_epoch_end/R2/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/on_epoch_end/R2/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], ```--lr``` เปลี่ยนเป็น 1e-5, เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ


### 💾 **[2.2] Train บนเครื่อง 29** 

- [x] **Transfer Learning**

```
# Train R1
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate AI
python3 train-Kfold.py --gpu 1 --R 1 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name transfer --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/tohn/HDD2/Model_unlearn/ResNet152v2Model/baseML_unlearn/R2/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_Unlearning_miniImageNet-R2.json --checkpoint_dir /media/tohn/HDD2/Model_unlearn/ResNet152v2Model/baseML_unlearn/R2/unfreeze_conv3_block-conv5_block/models/modelResNet152v2_Unlearning_miniImageNet-R2.weights.h5 \
        --weight imagenet --imgsize 224 --fold 5

# Resume R1 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 1 --R 1 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name transfer --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/tohn/HDD2/Model_unlearn/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/tohn/HDD2/Model_unlearn/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --imgsize 224 --fold 5 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม /path/to/model/modelname.json ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10]
    
- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R1 แบบ **resume**
- [ ]   ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/on_epoch_end/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/on_epoch_end/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ


- [x] **Fine-Tuning**
      
```
# Train R2
cd Machine-Unlearning-USAI-CCA-2026/USAI_unlearn
conda activate AI
python3 train-Kfold.py --gpu 1 --R 2 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name unfreeze_conv3_block-conv5_block --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/tohn/HDD2/Model_unlearn/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/models/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/tohn/HDD2/Model_unlearn/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R1_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_transfer_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --imgsize 224 --fold 5 --lr 1e-5

# Resume R2 - Load the Model to Continue Training to Completion.
python3 train-Kfold.py --gpu 1 --R 2 --network_name ResNet152v2 --set MLunlearn_USAI --data unbalanced --name unfreeze_conv3_block-conv5_block --exp unfreeze_conv3_block-conv5_block --Modeljson_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R2_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv3_block-conv5_block-R1_unbalanced_fold5_last.json --checkpoint_dir /media/HDD/mini-ImageNet/ResNet152v2Model/MLunlearn_USAI/run_Kfold/R2_unbalanced/transfer_exp_unfreeze_conv3_block-conv5_block/fold5/on_epoch_end/modelResNet152v2_MLunlearn_USAI_unfreeze_conv3_block-conv5_block_exp_unfreeze_conv3_block-conv5_block-R2_unbalanced_fold5_last.weights.h5 \
        --weight imagenet --imgsize 224 --fold 5 --lr 1e-5 --resume --epochendName on_epoch_end_resume --epochs 120
```

- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2
- [ ]  ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/model/R1/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/model/R1/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], ```--lr``` เปลี่ยนเป็น 1e-5
    
- [x] ⚙️ Parameter ที่ต้อง Set สำหรับ เทรน R2 แบบ **resume**
- [ ]   ```--gpu ``` เปลี่ยน [0, 1], ```--exp``` เปลี่ยน [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block], ```--Modeljson_dir``` เปลี่ยนตาม ```/path/to/on_epoch_end/R2/modelname.json```, ```--checkpoint_dir``` เปลี่ยน ตาม path ใน  ```/path/to/on_epoch_end/R2/modelname.weights.h5``` ให้ related กับ ```--exp``` , ```--fold``` เปลี่ยน [3, 4, 5, 6, 7, 8, 9, 10], ```--lr``` เปลี่ยนเป็น 1e-5, เพิ่ม ```--resume```, ```--epochendName``` เปลี่ยนชื่อเป็น on_epoch_end_resume, ```--epochs``` เปลี่ยน ตามจำนวน Epoch ที่ต้องเทรนต่อให้จบ

      
-------------------------------------------------------------------------------------------------------------------------------------------------------

### 💡 [3] Train ViT-L32 Network

- เทรน ```ViT-L32``` ทั้งหมดสามารถเทรนบนเครื่อง 28 ได้เลย เพราะโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` อยู่ที่เครื่อง 28 ทั้งหมด
- ลักษณะการเทรนจะเป็นการนำโมเดล Base ที่เทรนด้วย ```mini-ImageNet``` ด้วยแบบต่างๆ (มีแบบเดียว) ```[Transformer  & FC layers]``` เอามา Fine-Tune บนชุดข้อมูล USAI15AB แบบ 8-fold Validation

- [x] **Fine-Tuning**
      


