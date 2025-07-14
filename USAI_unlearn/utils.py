import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
import pandas as pd
from EffNetmodels import loadresumemodel, loadmodelUnlearn, finetuneUSAI_B4, build_EffNetmodelB5, model_block5Unfreze, finetuneUSAI_B4ToB7, finetuneUSAI_B1ToB4, finetuneUSAI_B5ToB7
from ResNet152v2Model import build_baseResNet152v2, ResNetUnfreeze_conv3_block, ResNetUnfreeze_conv3to5_block, ResNetUnfreeze_conv1xto3_block, loadresumemodel_ResNet
from ResNet152v2Model import CreateResNet152v2modelUnlearn
from data_loader import Data_generator
from tensorflow.keras.models import load_model




def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']




def utils_createModel(network_name, sets, R, name, exp, weight, resume=False, checkpoint_dir=None, Modeljson_dir=None, imgsize=None):
    # set up weight
    if weight == 'random':
        init_weight = None
    else:
        init_weight = 'imagenet'
    ## Create Model
    if network_name == "EffNetB5":
        if sets == "MLunlearn_USAI": 
            if resume :
                input_shape, model = loadresumemodel(checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print(colorstr('blue', f"----- [INFO]: Load ML Unlearn {exp} Model with {weight} weight to Finetune Stage: Transfer Learning Stage [Unfreeze FC layer]-----"))
                input_shape, model = loadmodelUnlearn(checkpoint_dir)
            elif R == 2 and name == "unfreezeB4" :
                print(colorstr('blue', f"[INFO]: Load ML Unlearn {exp} Model to Finetune Stage: Unfreeze Block4"))
                input_shape, model = finetuneUSAI_B4(Modeljson_dir, checkpoint_dir)
            elif R == 2 and name == "unfreezeB4-B7" :
                print(colorstr('blue', f"[INFO]: Load ML Unlearn {exp} Model to Finetune Stage: Unfreeze Block4 to Block7"))
                input_shape, model = finetuneUSAI_B4ToB7(Modeljson_dir, checkpoint_dir) 
            elif R == 2 and name == "unfreezeB1-B4" :
                print(colorstr('blue', f"[INFO]: Load ML Unlearn {exp} Model to Finetune Stage: Unfreeze Block1 to Block4"))
                input_shape, model = finetuneUSAI_B1ToB4(Modeljson_dir, checkpoint_dir)
            elif R == 2 and name == "unfreezeBlock5a_se_excite" :
                print(colorstr('blue', f"[INFO]: Load ML Unlearn {exp} Model to Finetune Stage: Unfreeze Block5a_se_excite to Block7"))
                input_shape, model = finetuneUSAI_B5ToB7(Modeljson_dir, checkpoint_dir)
        elif sets == "MLorigin_USAI": 
            ## Create Model
            if resume :
                input_shape, model = loadresumemodel(checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print(colorstr('blue', f'[INFO]: Build EffNetB5 Base Model with {weight} weight to Transfer Learning Stage'))
                input_shape, model = build_EffNetmodelB5(fine_tune=True, Numclasses=15, init_weight=init_weight)
            elif R == 2 and name == "unfreezeBlock5a_se_excite" :
                print(colorstr('blue', f"[INFO]: Load EffNetB5 {weight} Weight Model to Finetune Stage: Unfreeze Block5a_se_excite Layer"))
                input_shape, model = model_block5Unfreze(Modeljson_dir, checkpoint_dir)
    ## Create ResNet152V2 Model
    elif network_name == "ResNet152v2":
        if sets == "MLorigin_USAI": 
            if resume :
                print(f"---------- [INFO]: Load ResNet152V2 Base Model to Resume Training R{R} Stage----------")
                input_shape, model = loadresumemodel_ResNet(Modeljson_dir, checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print(colorstr('blue', f"----- [INFO]: Build ResNet152V2 Base Model with {weight} weight to Transfer Learning Stage -----"))
                input_shape, model = build_baseResNet152v2(imagesize=imgsize, init_weight=init_weight, Numclasses=15)
            elif R == 2 and name == "unfreeze_conv3_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv3_block ----------")
                input_shape, model = ResNetUnfreeze_conv3_block(Modeljson_dir, checkpoint_dir, sets)
            elif R == 2 and name == "unfreeze_conv3_block-conv5_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv3_block - conv5_block ----------")
                input_shape, model = ResNetUnfreeze_conv3to5_block(Modeljson_dir, checkpoint_dir, sets)
            elif R == 2 and name == "unfreeze_conv1-conv3_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv1x - conv3_block ----------")
                input_shape, model = ResNetUnfreeze_conv1xto3_block(Modeljson_dir, checkpoint_dir, sets)
        elif sets == "MLunlearn_USAI": 
             if resume :
                 (f"---------- [INFO]: Load ResNet152V2 Unlearn Model to Resume Training R{R} Stage, from {name} network ----------")
                 input_shape, model = loadresumemodel_ResNet(Modeljson_dir, checkpoint_dir)
             elif R == 1 and name == "transfer" :
                  print(f"---------- [INFO]: Create ResNet152V2 Unlearn Model to Transfer Learning Stage, from {exp} network ----------")
                  input_shape, model = CreateResNet152v2modelUnlearn(Modeljson_dir, checkpoint_dir, imgsize, Numclasses=15)
             elif R == 2 and name == "unfreeze_conv3_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv3_block ----------")
                  input_shape, model = ResNetUnfreeze_conv3_block(Modeljson_dir, checkpoint_dir, sets)
             elif R == 2 and name == "unfreeze_conv3_block-conv5_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv3_block - conv5_block ----------")
                  input_shape, model = ResNetUnfreeze_conv3to5_block(Modeljson_dir, checkpoint_dir, sets)
             elif R == 2 and name == "unfreeze_conv1-conv3_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv1x - conv3_block ----------")
                  input_shape, model = ResNetUnfreeze_conv1xto3_block(Modeljson_dir, checkpoint_dir, sets)
                
    return input_shape, model




    