"""Main script for project."""
from __future__ import print_function
import config
import data_loader
import end2end
import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import sys
#from model_components0 import maskCNNModel0, classificationHybridModel0
from model_components00 import maskCNNModel0, classificationHybridModel0
from model_components1 import maskCNNModel1, classificationHybridModel1
from model_components2 import maskCNNModel2, classificationHybridModel2
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(opts, maskCNNModel, classificationHybridModel):
    maskCNN_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')
    C_XtoY_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')
    print('LOAD MODEL:', maskCNN_path)

    maskCNN = maskCNNModel(opts)
    state_dict = torch.load(maskCNN_path, map_location=lambda storage, loc: storage)
    for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
    #state_dict['conv2.1.weight']= torch.cat((state_dict['conv2.1.weight'], torch.zeros(64,258-130,5,5)),1)
    #state_dict['fc1.weight']= torch.cat((state_dict['conv3.1.weight'], torch.zeros(64,258-130,5,5)),1)
    maskCNN.load_state_dict(state_dict)#, strict=False)

    C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel,
                                       conv_dim_out=opts.n_classes,
                                       conv_dim_lstm=opts.cxtoy_conv_dim_lstm)
    state_dict = torch.load( C_XtoY_path, map_location=lambda storage, loc: storage)
    for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
    #state_dict['dense.weight']= state_dict['dense.weight'][:,:state_dict['dense.weight'].shape[1]//opts.stack_imgs ]
    C_XtoY.load_state_dict(state_dict)#, strict=False)
    return maskCNN, C_XtoY


def main(opts,mask_CNN, C_XtoY):
    torch.cuda.empty_cache()

    # Create train and test dataloaders for images from the two domains X and Y
    training_dataloader, val_dataloader, testing_dataloader = data_loader.lora_loader(opts)
    # Create checkpoint directories

    # Start training
    set_gpu(opts.free_gpu_id)

    # start training
    mask_CNN, C_XtoY = end2end.training_loop(training_dataloader,val_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts)
    return mask_CNN, C_XtoY

if __name__ == "__main__":
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    print('COMMAND:    ', ' '.join(sys.argv))
    parser = config.create_parser()
    opts = parser.parse_args()

    if opts.SpFD =='True': opts.fs =opts.fs // opts.stack_imgs
    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw
    opts.stft_window = opts.n_classes // 2 
    opts.stft_overlap = opts.stft_window // 2
    opts.conv_dim_lstm = opts.n_classes * 8
    opts.freq_size = opts.n_classes
    opts.checkpoint_dir += 'M'+str(opts.model_ver)
    create_dir(opts.checkpoint_dir)
    opts.cxtoy_conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    if opts.SpFD =='True': opts.cxtoy_conv_dim_lstm *= opts.stack_imgs 

    if len(opts.snr_list)<opts.stack_imgs: opts.snr_list = [opts.snr_list[0] for i in range(opts.stack_imgs)]
    if opts.load_checkpoint_dir == '/data/djl':
        opts.load_checkpoint_dir = opts.checkpoint_dir
    if opts.data_dir == '/data/djl/data0306/data': opts.data_format = 0
    elif opts.data_dir == '/data/djl/SpF102': opts.data_format = 1
    elif opts.data_dir == '/data/djl/sf8_76800' or opts.data_dir == '/data/djl/sf9_76800' or '_125k_new' in opts.data_dir: opts.data_format = 2
    elif opts.data_dir == '/data/djl/sf7-1b-out-upload': opts.data_format = 3
    else: raise NotImplementedError
    

    
    ##load model checkpoint
    if opts.model_ver == 0:
        maskCNNModel = maskCNNModel00
        classificationHybridModel = classificationHybridModel00
    elif opts.model_ver == 1:
        maskCNNModel = maskCNNModel1
        classificationHybridModel = classificationHybridModel1
    elif opts.model_ver == 2:
        maskCNNModel = maskCNNModel2
        classificationHybridModel = classificationHybridModel2
    else: raise ValueError('Unknown Model Version')

    if opts.load == 'yes':
        if opts.load_iters == -1:
            vals = [int(fname.split('_')[0]) for fname in os.listdir(opts.load_checkpoint_dir) if fname[-4:] == '.pkl']
            if len(vals)==0: 
                opts.load = 'no'
                print('--WARNING: CHECKPOINT_DIR NOT EXIST, SETTING OPTS.LOAD TO NO--')
            else: opts.load_iters = max(vals)
    if opts.load == 'yes':
        print('LOAD ITER:  ',opts.load_iters)
        mask_CNN, C_XtoY = load_checkpoint(opts, maskCNNModel, classificationHybridModel)
    else:
        mask_CNN = maskCNNModel(opts)
        C_XtoY = classificationHybridModel(conv_dim_in=opts.y_image_channel, conv_dim_out=opts.n_classes, conv_dim_lstm= opts.cxtoy_conv_dim_lstm)
    mask_CNN = nn.DataParallel(mask_CNN)
    C_XtoY = nn.DataParallel(C_XtoY)

    mask_CNN.cuda()
    C_XtoY.cuda()
    
    #Loads the data, creates checkpoint and sample directories, and starts the training loop.


    opts.logfile = os.path.join(opts.checkpoint_dir, 'logfile-djl-train.txt')
    opts.logfile2 = os.path.join(opts.checkpoint_dir, 'logfile2-djl-train.txt')
    strlist = print_opts(opts)
    with open(opts.logfile,'a') as f: f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'\n'.join(strlist)+'\n')
    with open(opts.logfile2,'a') as f: f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'\n'.join(strlist)+'\n')
    opts.init_train_iter = opts.load_iters
    mask_CNN, C_XtoY = main(opts,mask_CNN, C_XtoY)
    opts.init_train_iter += opts.train_iters

