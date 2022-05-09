# end2end.py

from __future__ import division
import os
import sys
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy.io

import cv2
# Local imports
from utils import to_var, to_data, spec_to_network_input, create_dir
from model_components import maskCNNModel, classificationHybridModel
import torch.autograd.profiler as profiler
import time

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)





def checkpoint(iteration, mask_CNN, C_XtoY, opts):
    mask_CNN_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_maskCNN.pkl')
    create_dir(opts.checkpoint_dir)
    torch.save(mask_CNN.state_dict(), mask_CNN_path)
    C_XtoY_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_C_XtoY.pkl')
    torch.save(C_XtoY.state_dict(), C_XtoY_path)
    print('model checkpoint saved to', mask_CNN_path, C_XtoY_path)




def training_loop(training_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
    """
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = nn.CrossEntropyLoss()
    
    # Create generators and discriminators
    g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])

    # Maintain Log of average model loss of latest opts.log_step*5 steps
    logfile = os.path.join(opts.log_dir, 'log' + str(opts.snr_list[0])+'_'+str(opts.snr_list[-1])+'_'+str(opts.stack_imgs) + '.txt')
    G_Y_loss_avg = []
    G_Image_loss_avg = []
    G_Class_loss_avg = []

    iteration = 0
    oldtime = time.time()#time the training process

    while iteration < opts.train_iters:
        train_iter = iter(training_dataloader)
        print('start new training epoch')
        for images_X, labels_X, images_Y in train_iter:
            if iteration>opts.train_iters:break
            iteration+=1

            images_X = to_var(images_X)
            images_Y = to_var(torch.tensor(images_Y, dtype=torch.cfloat))

            # ============================================
            #            GENERATE TRAIING IMAGES
            # ============================================
            images_X_spectrum = [] # training images * opts.stack_imgs
            for i in range(opts.stack_imgs):
                images_X_spectrum_raw = torch.stft(input=images_X.select(1,i), n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                                   win_length=opts.stft_window, pad_mode='constant');
                images_X_spectrum.append( spec_to_network_input(images_X_spectrum_raw, opts) )
            
            images_Y_spectrum_raw = torch.stft(input=images_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                               win_length=opts.stft_window, pad_mode='constant');
            images_Y_spectrum = spec_to_network_input(images_Y_spectrum_raw, opts) 
            
            #########################################
            ##              FORWARD
            #########################################
            fake_Y_spectrum = mask_CNN(images_X_spectrum) #CNN input: a list of images, output: a list of images
            fake_Y_spectrum = torch.mean(torch.stack(fake_Y_spectrum,0),0) #average of CNN outputs
            
            g_y_pix_loss = loss_spec(fake_Y_spectrum, images_Y_spectrum)
            labels_X_estimated = C_XtoY(fake_Y_spectrum)
            g_y_class_loss = loss_class(labels_X_estimated, labels_X)
            g_optimizer.zero_grad()
            G_Image_loss = opts.scaling_for_imaging_loss * g_y_pix_loss
            G_Class_loss = opts.scaling_for_classification_loss * g_y_class_loss
            G_Y_loss = G_Image_loss + G_Class_loss 
            G_Y_loss.backward()
            g_optimizer.step()

            
            #########################################
            ##     LOG THE LOSS OF LATEST opts.log_step*5 STEPS
            #########################################
            if len(G_Y_loss_avg)<opts.log_step*5:
                G_Y_loss_avg.append(G_Y_loss.item())
                G_Image_loss_avg.append(G_Image_loss.item())
                G_Class_loss_avg.append(G_Class_loss.item())
            else:
                G_Y_loss_avg[iteration % opts.log_step*5] = G_Y_loss.item()
                G_Image_loss_avg[iteration % opts.log_step*5] = G_Image_loss.item()
                G_Class_loss_avg[iteration % opts.log_step*5] = G_Class_loss.item()
            if iteration % opts.log_step == 0:
                output_str = 'Train Iteration [{:6d}/{:5d}] | G_Y_loss: {:6.4f}| G_Image_loss: {:6.4f}| G_Class_loss: {:6.4f} | Time: {:.2f}' .format(iteration,opts.train_iters,
                                np.mean(G_Y_loss_avg),
                                np.mean(G_Image_loss_avg),
                                np.mean(G_Class_loss_avg),
                                time.time() - oldtime)
                oldtime = time.time()
                print(output_str)

            ## checkpoint
            if (iteration+1) % opts.checkpoint_every == 0:
                checkpoint(iteration, mask_CNN, C_XtoY, opts)

            ## test
            if iteration % opts.test_step == 1 or iteration == opts.train_iters:
                print('start testing..')
                error_matrix = 0
                error_matrix_count = 0
                test_iter = iter(testing_dataloader)
                for images_X_test, labels_X_test, images_Y_test in test_iter:

                        #prepare testing data
                        images_X_test, labels_X_test = to_var(images_X_test), to_var(labels_X_test)
                        images_Y_test = to_var(images_Y_test)
                        images_X_test_spectrum = []
                        for i in range(opts.stack_imgs):
                            images_X_test_spectrum_raw = torch.stft(input=images_X_test.select(1,i), n_fft=opts.stft_nfft,
                                                                    hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                                    pad_mode='constant');
                            images_X_test_spectrum.append(spec_to_network_input(images_X_test_spectrum_raw, opts))

                        images_Y_test_spectrum_raw = torch.stft(input=images_Y_test, n_fft=opts.stft_nfft,
                                                                hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                                pad_mode='constant');
                        images_Y_test_spectrum = spec_to_network_input(images_Y_test_spectrum_raw, opts)

                        # forward
                        fake_Y_test_spectrums = mask_CNN(images_X_test_spectrum)
                        fake_Y_test_spectrum = torch.mean(torch.stack(fake_Y_test_spectrums,0),0)
                        labels_X_estimated = C_XtoY(fake_Y_test_spectrum)

                        #get the answer
                        _, labels_X_test_estimated = torch.max(labels_X_estimated, 1)
                        test_right_case = (labels_X_test_estimated == labels_X_test)
                        test_right_case = to_data(test_right_case)
                        error_matrix += np.sum(test_right_case)
                        error_matrix_count += opts.batch_size
                error_matrix = error_matrix / error_matrix_count
                print('test accuracy',error_matrix,'logged to', logfile)
                with open(logfile,'a') as f:
                    f.write(str(iteration) +  ' ' + str(error_matrix)+'\n')
    return mask_CNN, C_XtoY
