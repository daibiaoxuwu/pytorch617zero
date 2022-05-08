"""Main script for project."""
from __future__ import print_function
import config
import datasets.data_loader as data_loader
import end2end
import os
import pickle
import numpy as np
import random
import torch
from models.model_components import maskCNNModel, classificationHybridModel
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(opts):
    maskCNN_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')
    C_XtoY_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')
    print('loading model checkpoint from', C_XtoY_path, maskCNN_path)

    maskCNN = maskCNNModel(opts)
    maskCNN.load_state_dict(torch.load(
        maskCNN_path, map_location=lambda storage, loc: storage),
        strict=False)

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

def main(opts,files_train,files_test,mask_CNN, C_XtoY):
    torch.cuda.empty_cache()

    # Create train and test dataloaders for images from the two domains X and Y
    training_dataloader, testing_dataloader = data_loader.lora_loader(
        opts, files_train, files_test)
    # Create checkpoint directories
    create_dir(opts.checkpoint_dir)

    # Start training
    set_gpu(opts.free_gpu_id)

    # start training
    mask_CNN, C_XtoY = end2end.training_loop(training_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts)
    return mask_CNN, C_XtoY

if __name__ == "__main__":
    parser = config.create_parser()
    opts = parser.parse_args()

    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw
    opts.stft_window = opts.n_classes // 2
    opts.stft_overlap = opts.stft_window // 2
    opts.conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    opts.freq_size = opts.n_classes


    ##load model checkpoint
    if opts.load == 'yes':
        print('load checkpoint')
        mask_CNN, C_XtoY = load_checkpoint(opts)
    else:
        mask_CNN, C_XtoY = create_model(opts)
    
    #Loads the data, creates checkpoint and sample directories, and starts the training loop.

    Y_filenames = []
    for i in range(128):
        filelist = os.listdir(os.path.join(opts.data_dir,str(i),str(opts.groundtruth_code))) #use SNR<-15 for training data(X); SNR=+35 for groundtruth(Y). we first find the SNR=+35 data (Y) and then read the corresponding SNR<-15 data for training(X).
        Y_filenames.extend([ os.path.join(opts.data_dir,str(i),str(opts.groundtruth_code),j) for j in filelist])
    random.shuffle(Y_filenames)

    num_files = len(Y_filenames)
    num_train = int(num_files * opts.ratio_bt_train_and_test)
    files_train = Y_filenames[0:num_train]
    files_test = Y_filenames[num_train:num_files]

    print("length of training and testing data is {},{}".format(len(files_train), len(files_test)))
    print('start training with snr',opts.snr_list,'stack',opts.stack_imgs)
            
    mask_CNN, C_XtoY = main(opts,files_train, files_test,mask_CNN, C_XtoY)

