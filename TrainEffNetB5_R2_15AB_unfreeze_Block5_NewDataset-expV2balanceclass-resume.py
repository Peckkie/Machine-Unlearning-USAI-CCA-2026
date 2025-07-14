import PIL
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.models import load_model



os.environ["CUDA_VISIBLE_DEVICES"]="1"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 16
epochs = 71

#Train
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Traindf_fold4_8_v1_Balance_classes.csv')
#validation
valframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Valdf_fold3_v1.csv')
#from efficientnet.keras import EfficientNetB5 as Net
model_dir = "/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/on_epoch_end/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2_last.h5"
model = load_model(model_dir)
print(f"Load model to resume: {model_dir}")
print("="*125)
height = width = model.input_shape[1] 
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

## Create path to save model
root_model = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/models'
os.makedirs(root_model, exist_ok=True)
## Create path to save tensorboard
root_logdir = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/Mylogs_tensor'
os.makedirs(root_logdir, exist_ok=True)
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)
### Run TensorBoard 
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
   
root_Metrics = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/on_epoch_end_resume'
os.makedirs(root_Metrics, exist_ok=True)
class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                if epochs%20 == 0 and epochs != 0:
                    self.model.save(f'{root_Metrics}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2_epoch{epochs}.h5')
                else:
                    self.model.save(f'{root_Metrics}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2_last.h5')
                return

# For tracking Quadratic Weighted Kappa score and saving best weights
metrics = Metrics()

## Compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['acc'])

# Fit model with class weights
model.fit(train_generator, epochs=epochs, 
              validation_data=test_generator,
              callbacks=[metrics, tensorboard_cb])


model.save(f'{root_model}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2.h5')
print(f"=============== [INFO]: Save Model Completed >>> {root_model}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2.h5 ===============")

