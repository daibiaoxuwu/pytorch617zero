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
import sys
from model_components import maskCNNModel, classificationHybridModel
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(opts):
    maskCNN_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')
    C_XtoY_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')
    print('loading model checkpoint from', C_XtoY_path, maskCNN_path)

    maskCNN = maskCNNModel(opts)
    state_dict = torch.load(maskCNN_path, map_location=lambda storage, loc: storage)
    #state_dict['conv2.1.weight']= torch.cat((state_dict['conv2.1.weight'], torch.zeros(64,258-130,5,5)),1)
    #state_dict['conv3.1.weight']= torch.cat((state_dict['conv3.1.weight'], torch.zeros(64,258-130,5,5)),1)
    maskCNN.load_state_dict(state_dict, strict=True)

    C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel,
                                       conv_dim_out=opts.n_classes,
                                       conv_dim_lstm=opts.conv_dim_lstm)
    C_XtoY.load_state_dict(torch.load(
        C_XtoY_path, map_location=lambda storage, loc: storage),
        strict=False)
    if torch.cuda.is_available():
        maskCNN.cuda()
        C_XtoY.cuda()
    return maskCNN, C_XtoY


def main(opts,mask_CNN, C_XtoY):
    torch.cuda.empty_cache()

    # Create train and test dataloaders for images from the two domains X and Y
    training_dataloader, testing_dataloader = data_loader.lora_loader(opts)
    # Create checkpoint directories
    create_dir(opts.checkpoint_dir)

    # Start training
    set_gpu(opts.free_gpu_id)

    # start training
    mask_CNN, C_XtoY = end2end.training_loop(training_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts)
    return mask_CNN, C_XtoY

if __name__ == "__main__":
    print(sys.argv)
    parser = config.create_parser()
    opts = parser.parse_args()

    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw
    opts.stft_window = opts.n_classes // 2
    opts.stft_overlap = opts.stft_window // 2
    opts.conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    opts.freq_size = opts.n_classes


    ##load model checkpoint
    if opts.load_iters == -1:
        vals = [int(fname.split('_')[0]) for fname in os.listdir(opts.load_checkpoint_dir)]
        opts.load_iters = max(vals)
        print('load newest iteration:',opts.load_iters)
    if opts.load == 'yes':
        mask_CNN, C_XtoY = load_checkpoint(opts)
    else:
        mask_CNN = maskCNNModel(opts)
        C_XtoY = classificationHybridModel(conv_dim_in=opts.y_image_channel,
                                           conv_dim_out=opts.n_classes,
                                           conv_dim_lstm=opts.conv_dim_lstm)
        if torch.cuda.is_available():
            mask_CNN.cuda()
            C_XtoY.cuda()
    
    #Loads the data, creates checkpoint and sample directories, and starts the training loop.


    opts.logfile = os.path.join(opts.log_dir, 'logtest-djl-train.txt')
    with open(opts.logfile,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +str(sys.argv)+'\n')
    opts.init_train_iter = opts.load_iters+1
    for snr in range(opts.snr_list[0]-1,-26,-1):
        opts.snr_list = [snr,]
        print('start training with snr',opts.snr_list,'stack',opts.stack_imgs)
        with open(opts.logfile,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' snr ' + str(opts.snr_list) + ' stack ' + str(opts.stack_imgs) + '\n')
        mask_CNN, C_XtoY = main(opts,mask_CNN, C_XtoY)
        opts.init_train_iter += opts.train_iters

