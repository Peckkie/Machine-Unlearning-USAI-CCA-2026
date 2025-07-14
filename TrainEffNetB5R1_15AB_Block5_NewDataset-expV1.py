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
import os
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from efficientnet.keras import *
#from tensorflow.keras.optimizers import legacy

# Disable eager execution
#tf.compat.v1.disable_eager_execution()
### Set GPU env.
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 16
epochs = 200

#Train
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Traindf_fold4_8_v1.csv')
base_dir = '/media/tohn/SSD/Images/Image1'
os.chdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
#validation
valframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Valdf_fold3_v1.csv')
validation_dir = os.path.join(base_dir, 'validation')

#from efficientnet.keras import EfficientNetB5 as Net
from tensorflow.keras.applications import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
height = width = 456
input_shape = (height, width, 3)
# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)
# create new model with a new classification layer
x = conv_base.output  
global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
prediction_layer = layers.Dense(15, activation='softmax',name = 'prediction_layer')(dropout_layer_1)

model = models.Model(inputs= conv_base.input, outputs=prediction_layer) 

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
#conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
for layer in conv_base.layers:
    layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense
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
        directory = train_dir,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

## Create path to save model
root_model = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV1/R1/models'
os.makedirs(root_model, exist_ok=True)
## Create path to save tensorboard
root_logdir = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV1/R1/Mylogs_tensor'
os.makedirs(root_logdir, exist_ok=True)
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)
### Run TensorBoard 
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
# Set up ModelCheckpoint to save only on certain epochs
root_Metrics = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV1/R1/on_epoch_end'
os.makedirs(root_Metrics, exist_ok=True)  

class SaveModelOnSpecificEpoch(tf.keras.callbacks.Callback):
    def __init__(self, root_Metrics):
        # Import TensorFlow here
        import tensorflow as tf
        self.root_Metrics = root_Metrics

    def on_epoch_end(self, epochs, logs=None):
        # Save only every 20 epochs, excluding the 0th epoch
        if epochs % 20 == 0 and epochs != 0:
            self.model.save(  # Corrected method name
                f"{self.root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-epoch{epochs}.h5"
            )
        else:
            self.model.save(  # Corrected method name
                f"{self.root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-last.h5"
            )
            
# class SaveModelOnSpecificEpoch(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epochs, logs=None):
#         #import tensorflow as tf  # Import TensorFlow here
#         # Save only every 20 epochs, excluding the 0th epoch
#         if epochs % 20 == 0 and epochs != 0:
#             #self.model.save(f"{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-epoch{epochs}.h5")
#             self.tf.keras.models.save_model(model, f'{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-epoch{epochs}.h5')
#         else:
#             #self.model.save(f"{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-last.h5")
#             self.tf.keras.models.save_model(model, f'{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-last.h5')
#         return
            
# Instantiate the custom callback
save_callback = SaveModelOnSpecificEpoch(root_Metrics)            
                      
# checkpoint_path = f"{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-last.h5"
# checkpoint_cb = ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=False,
#     save_best_only=False,
#     save_freq='epoch',  # Saves every epoch
# )  
  
#Remove `period`, handle frequency of saving by filtering epochs in the callback
# def save_only_on_specific_epochs(epochs, logs):
#     # Save only every 20 epochs
#     if epochs % 20 == 0 and epochs != 0:
#         model.save(f"{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-epoch{epochs}.h5")
#     return

# custom_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_only_on_specific_epochs)

# Custom callback to save model every 20 epochs
# custom_callback = tf.keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epochs, logs: model.save(f"{root_Metrics}/modelEffnetB5R1_15AB_fold4_8_NewDataset-epoch{epochs}.h5") 
#     if epochs % 20 == 0 and epochs != 0 else None
    
# def avoid_error(gen):
#     while True:
#         try:
#             data, labels = next(gen)
#             yield data, labels
#         except:
#             pass

#Training
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['acc'])


## Fit model 
model.fit(train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks = [tensorboard_cb, save_callback])

model.save(f'{root_model}/modelEffnetB5R1_15AB_fold4_8_NewDataset.h5')
# Save the model
#tf.keras.models.save_model(model, f'{root_model}/modelEffnetB5R1_15AB_fold4_8_NewDataset.h5')


print(f"=============== [INFO]: Save Model Completed >>> {root_model}/modelEffnetB5R1_15AB_fold4_8_NewDataset.h5 ===============")


