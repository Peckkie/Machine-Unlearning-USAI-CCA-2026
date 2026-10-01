import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
# from keras.callbacks import Callback
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from utils import utils_createModel
from data_loader import Data_generator
#from efficientnet.keras import EfficientNetB5 as Net
#load Check point
from tensorflow.keras.models import load_model
from utils import colorstr
import argparse
import random



def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


class Metrics(Callback):
    def __init__(self, root_metrics, on_epoch_name):
        super(Metrics, self).__init__()
        self.root_metrics = root_metrics
        self.on_epoch_name = on_epoch_name

    def on_epoch_end(self, epochs, logs=None):
        if logs is None:
            logs = {}
        # Check if this is a checkpoint epoch (every 50 epochs)
        if epochs % 20 == 0 and epochs != 0:
            # Save model, weights, and JSON at the checkpoint epoch
            self.model.save(f'{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.h5')
            self.model.save_weights(f'{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.weights.h5')
            # Serialize model to JSON
            model_json = self.model.to_json()
            with open(f"{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.json", "w") as json_file:
                json_file.write(model_json)
        
        else:
            # Save the latest model and weights after each epoch
            self.model.save(f'{self.root_metrics}{self.on_epoch_name}_last.h5')
            self.model.save_weights(f'{self.root_metrics}{self.on_epoch_name}_last.weights.h5')
            # Serialize the latest model to JSON
            model_json = self.model.to_json()
            with open(f"{self.root_metrics}{self.on_epoch_name}_last.json", "w") as json_file:
                json_file.write(model_json)





def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    my_parser.add_argument('--network_name', type=str, default='ResNet152v2', help='[ResNet152v2, EffNetB5]')
    my_parser.add_argument('--weight', type=str ,default='.', help='[imagenet, random]')
    my_parser.add_argument('--set', type=str, default='MLunlearn_USAI', help='In this work, For finetuning with USAI 15 AB [MLunlearn_USAI, MLorigin_USAI]')
    my_parser.add_argument('--data_path', type=str, default='/media/tohn/HDD/VISION_dataset')
    my_parser.add_argument('--data', type=str, default='balanced', help='[balanced, unbalanced]')
    my_parser.add_argument('--save_dir', type=str, default="/media/tohn/HDD2/Model_unlearn", help='Main Output Path >> [/media/tohn/HDD2/Model_unlearn/{network_name}Model]')
    my_parser.add_argument('--imgsize', type=int, default=456, help='[EffNetB5: 456, ResNet152v2: 224]')
    my_parser.add_argument('--name', type=str, default=".", help='EffNetB5: [transfer, unfreezeB4, unfreezeB1-B4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ResNet152v2: [transfer, unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block]')
    my_parser.add_argument('--exp', type=str, default=".", help='Only during Train R1: EffNetB5: [unfreezeB4, unfreezeB1-B4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ResNet152v2: [unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block]')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--fold', type=int, help='[Fold3 - Fold10]')
    my_parser.add_argument('--lr', type=float, default=2e-5)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
    my_parser.add_argument('--Modeljson_dir', type=str ,default=".")
    my_parser.add_argument('--tensorName', type=str ,default="Mylogs_tensor")
    #my_parser.add_argument('--checkpointerName', type=str ,default="checkpointer")
    my_parser.add_argument('--epochendName', type=str ,default="on_epoch_end")
    my_parser.add_argument('--FmodelsName', type=str ,default="models")
    my_parser.add_argument('--seed', type=int, default=None, help='X2 repeated runs: [1, 2, 3] ; default None = old behaviour (no seed, old output path)')
    my_parser.add_argument('--tag', type=str, default="", help='Control runs: [C1_computematched, C2_sameimg] ; saved in {set}_{tag} folder')
    
    args = my_parser.parse_args()

    ## set seed (X2 repeated runs)
    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(colorstr('yellow', f'[INFO]: Set random seed = {args.seed}'))
    
    ## set gpu
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## Create Model
    input_shape, model = utils_createModel(network_name=args.network_name, sets=args.set, weight=args.weight, resume=args.resume, R=args.R, name=args.name, exp=args.exp, checkpoint_dir=args.checkpoint_dir, Modeljson_dir=args.Modeljson_dir, imgsize=args.imgsize)       
    ## get images size 
    IMAGE_SIZE = input_shape[0]
    model.summary()
    print('='*100)
    
    # ## import dataset to create dataloader
    if args.data == "balanced":
       ### Train set balanced classes 
       dataframe = pd.read_csv(f'{args.data_path}/Traindf_fold4_8_v1_Balance_classes.csv')
    elif args.data == "unbalanced":
        ### Train set imbalanced classes  
        dataframe = pd.read_csv(f'{args.data_path}/Traindf_fold3_10_Kfold.csv')
    print(colorstr('green', f'[INFO]: Training Model on USAI Dataset ==> [ Fold{args.fold} ] <=='))
    ## train set
    trainframe = dataframe[dataframe["fold"]!=args.fold].reset_index(drop=True)
    ## validation set
    valframe = dataframe[dataframe["fold"]==args.fold].reset_index(drop=True)
    ### Implement > ## Train and validation sets  
    train_generator, valid_generator = Data_generator(height=IMAGE_SIZE, width=IMAGE_SIZE, BATCH_SIZE=args.batchsize, 
                                                          dataframe=trainframe, valframe=valframe, seed=args.seed)
    
    ## Create Main Output Path
    _R = f"R{args.R}"
    ## Control / repeated runs are saved in separate folders so existing results are not overwritten
    set_dir = f'{args.set}_{args.tag}' if args.tag else args.set
    run_tag = (f'_{args.tag}' if args.tag else '') + (f'_seed{args.seed}' if args.seed is not None else '')
    if args.set == "MLunlearn_USAI":
        if args.R == 1:
            root_base = f'{args.save_dir}/{args.network_name}Model/{set_dir}/run_Kfold/{_R}_{args.data}/{args.name}_exp_{args.exp}/fold{args.fold}'
        elif args.R == 2:
            root_base = f'{args.save_dir}/{args.network_name}Model/{set_dir}/run_Kfold/{_R}_{args.data}/{args.name}/exp_{args.exp}/fold{args.fold}'
    elif args.set == "MLorigin_USAI":
        root_base = f'{args.save_dir}/{args.network_name}Model/{set_dir}/weight_{args.weight}/{_R}_{args.data}/{args.name}'    
    if args.seed is not None:
        root_base = f'{root_base}/seed{args.seed}'
    os.makedirs(root_base, exist_ok=True)
    ## Set mkdir TensorBoard 
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    #tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    tensorboard_cb = TensorBoard(log_dir=run_logdir, write_graph=False)
    ## Create Model Folder 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    ## Set Model Name 
    if args.set == "MLunlearn_USAI":
        modelName = f'model{args.network_name}_{args.set}_{args.name}_exp_{args.exp}-{_R}_{args.data}_fold{args.fold}{run_tag}.h5'
        ## Set check point Name
        on_epochName = f'model{args.network_name}_{args.set}_{args.name}_exp_{args.exp}-{_R}_{args.data}_fold{args.fold}{run_tag}'
    elif args.set == "MLorigin_USAI":
        modelName = f'model{args.network_name}_{args.set}_{args.name}-{_R}_{args.data}{run_tag}.h5'
        ## Set check point Name
        on_epochName = f'model{args.network_name}_{args.set}_{args.name}-{_R}_{args.data}{run_tag}'
    ## Create save epoch end folder 
    Model2save = f'{modelNamemkdir}/{modelName}'
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True) 
    # Initialize the Metrics callback
    metrics_callback = Metrics(root_metrics=root_Metrics, on_epoch_name=on_epochName)
    
    #Compile model
    #Training
    model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=args.lr),
                          metrics=['accuracy'])

    ## Fit model 
    model.fit(train_generator, epochs=args.epochs, batch_size=args.batchsize, validation_data=valid_generator,
                 callbacks=[metrics_callback, tensorboard_cb])

    # Save models.
    # serialize model to JSON 
    model_json = model.to_json()
    with open(f"{modelNamemkdir}/{on_epochName}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{modelNamemkdir}/{on_epochName}.weights.h5")
    print("Saved model to disk")
    ### Save model to .h5 
    model.save(Model2save)
    ### print
    #print(f"Save USAI Model as: {modelNamemkdir}")
    print(colorstr('green', f'Save {args.network_name} USAI Model as: {modelNamemkdir}'))
    print(f"*"*150)
    

    
    
## Run Function 
if __name__ == '__main__':
    main()
    
    





