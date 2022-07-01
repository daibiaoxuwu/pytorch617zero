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
from model_components0 import maskCNNModel0, classificationHybridModel0
from model_components1 import maskCNNModel1, classificationHybridModel1
from model_components2 import maskCNNModel2, classificationHybridModel2
from model_components3 import maskCNNModel3, classificationHybridModel3
from utils import *

def load_checkpoint(opts, maskCNNModel, classificationHybridModel):
    maskCNN_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')

    C_XtoY_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')
    print('LOAD MODEL:', maskCNN_path)

    maskCNN = maskCNNModel(opts)
    state_dict = torch.load(maskCNN_path, map_location=lambda storage, loc: storage)
    for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
    #state_dict['conv2.1.weight']= torch.cat((state_dict['conv2.1.weight'], torch.zeros(64,258-130,5,5)),1)
    #state_dict['fc1.weight']= state_dict['fc1.weight'][:, :4096]
    #state_dict.pop('fc2.weight')
    #state_dict.pop('fc2.bias')
    maskCNN.load_state_dict(state_dict)#, strict=False)

    if opts.cxtoy == 'True':
        C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel, conv_dim_out=opts.n_classes, conv_dim_lstm=opts.conv_dim_lstm)
        if opts.load_cxtoy == 'True' and os.path.exists(C_XtoY_path):
            state_dict = torch.load( C_XtoY_path, map_location=lambda storage, loc: storage)
            for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
            #state_dict['dense.weight']= state_dict['dense.weight'][:,:state_dict['dense.weight'].shape[1]//opts.stack_imgs ]
            C_XtoY.load_state_dict(state_dict)#, strict=False)
        return [maskCNN, C_XtoY]
    else: return [maskCNN, ]


def main(opts,models):
    print('================TESTING VERSION================')
    print('================TESTING VERSION================')
    print('================TESTING VERSION================')
    torch.cuda.empty_cache()

    # Create train and test dataloaders for images from the two domains X and Y
    training_dataloader, testing_dataloader = data_loader.lora_loader(opts)
    # Create checkpoint directories

    # Start training
    set_gpu(opts.free_gpu_id)

    # start training
    models = end2end.training_loop(training_dataloader,testing_dataloader,models, opts)
    return models

if __name__ == "__main__":
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    print('COMMAND:    ', ' '.join(sys.argv))
    parser = config.create_parser()
    opts = parser.parse_args()

    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw

    opts.stft_window = opts.n_classes // 2 * 4
    opts.stft_overlap = opts.stft_window // 2 // 4
    opts.conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    opts.freq_size = opts.n_classes

    opts.checkpoint_dir += 'M'+str(opts.model_ver)
    create_dir(opts.checkpoint_dir)

    if len(opts.snr_list)<opts.stack_imgs: opts.snr_list = [opts.snr_list[0] for i in range(opts.stack_imgs)]
    
    if opts.lr == -1:
        opts.lr = 0.001
        if min(opts.snr_list) < -15: opts.lr *= 0.3
        if min(opts.snr_list) < -20: opts.lr /= 1.5
    if opts.w_image == -1:
        opts.w_image = 1
        if min(opts.snr_list) < -15: opts.w_image *= 4
        if min(opts.snr_list) < -20: opts.w_image *= 4


    #default checkpoint dir
    if opts.load_checkpoint_dir == '/data/djl': opts.load_checkpoint_dir = opts.checkpoint_dir

    
    ##load model checkpoint
    if opts.model_ver == 0:
        maskCNNModel = maskCNNModel0
        classificationHybridModel = classificationHybridModel0
    elif opts.model_ver == 1:
        maskCNNModel = maskCNNModel1
        classificationHybridModel = classificationHybridModel1
    elif opts.model_ver == 2:
        maskCNNModel = maskCNNModel2
        classificationHybridModel = classificationHybridModel2
    elif opts.model_ver == 3:
        maskCNNModel = maskCNNModel3
        classificationHybridModel = classificationHybridModel3
    else: raise ValueError('Unknown Model Version')

    if opts.load == 'yes':
        if opts.load_iters == -1:
            vals = [int(fname.split('_')[0]) for fname in os.listdir(opts.load_checkpoint_dir) if fname[-4:] == '.pkl']
            if len(vals)==0 or max(vals) == 0: 
                opts.load = 'no'
                print('--WARNING: CHECKPOINT_DIR NOT EXIST, SETTING OPTS.LOAD TO NO--')
            else: opts.load_iters = max(vals)
    if opts.load == 'yes':
        print('LOAD ITER:  ',opts.load_iters)
        models = load_checkpoint(opts, maskCNNModel, classificationHybridModel)
        mask_CNN = models[0]
        if opts.cxtoy == 'True': C_XtoY = models[1]
    else:
        mask_CNN = maskCNNModel(opts)
        if opts.cxtoy == 'True': C_XtoY = classificationHybridModel(conv_dim_in=opts.out_channel, conv_dim_out=opts.n_classes, conv_dim_lstm= opts.conv_dim_lstm)
    mask_CNN = nn.DataParallel(mask_CNN)
    mask_CNN.cuda()
    models = [mask_CNN, ]
    if opts.cxtoy == 'True':
        C_XtoY = nn.DataParallel(C_XtoY)
        C_XtoY.cuda()
        models.append(C_XtoY)
    
    opts.logfile = os.path.join(opts.checkpoint_dir, 'logfile-djl-train.txt')
    opts.logfile2 = os.path.join(opts.checkpoint_dir, 'logfile2-djl-train.txt')
    strlist = print_opts(opts)
    with open(opts.logfile,'a') as f: f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'\n'.join(strlist)+'\n')
    with open(opts.logfile2,'a') as f: f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' snr_list ' +str(opts.snr_list)+' stack '+str(opts.stack_imgs)+' ')
    opts.init_train_iter = opts.load_iters
    models = main(opts,models)
    opts.init_train_iter += opts.train_iters

