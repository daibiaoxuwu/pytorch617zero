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
import torch.autograd.profiler as profiler
import time
import math
from torchvision import transforms
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
    print('CKPT: ', mask_CNN_path)

def binary(y):
    y[abs(y)>0.5] = 1
    y[abs(y)<=0.5] = 0
    return y

def merge_images(sources, targets, Y, opts):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources[0].shape
    row = int(np.sqrt(opts.batch_size))
    column = math.ceil(opts.batch_size / row)
    merged = np.zeros([1, row * h * opts.stack_imgs, column * w * 4])
    for stack_idx in range(opts.stack_imgs):
        for idx, (s, t, y) in enumerate(zip(sources[stack_idx], targets[stack_idx], Y[stack_idx])):
            i = idx // column
            j = idx % column
            t = (t-np.min(t))/(np.max(t)-np.min(t))
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 4) * w:(j * 4 + 1) * w] = s[0]/2+0.5
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 4 + 1) * w:(j * 4 + 2) * w] = s[1]
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 4 + 2) * w:(j * 4 + 3) * w] = t
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 4 + 3) * w:(j * 4 + 4) * w] = y
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, fixed_X, mask_CNN, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    if opts.model_ver == 0: fake_Y = [mask_CNN(i) for i in fixed_X]
    else: fake_Y = mask_CNN(fixed_X)
    fixed_X = [to_data(i) for i in fixed_X]

    Y, fake_Y = [to_data(i) for i in fixed_Y], [to_data(i) for i in fake_Y]

    merged_all = merge_images(fixed_X, fake_Y, Y, opts)

    for i in range(1):
        path = os.path.join(opts.checkpoint_dir,
                            'sample-{:06d}-Y.png'.format(iteration+i))
        merged = merged_all[:, :, i] 
        merged = (merged - np.amin(merged)) / (np.amax(merged) - np.amin(merged)) * 255
        merged = cv2.flip(merged, 0)
        cv2.imwrite(path, merged)
    print('SAMPLE: {}'.format(path))


def training_loop(training_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
    """
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = nn.CrossEntropyLoss()
    
    # Create generators and discriminators
    g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])

    G_Y_loss_avg = []
    G_Image_loss_avg = []
    G_Class_loss_avg = []

    iteration = opts.init_train_iter
    oldtime = time.time()#time the training process

    while iteration < opts.init_train_iter + opts.train_iters:
        train_iter = iter(training_dataloader)
        #print('start new training epoch')
        for images_X, labels_X, images_Y in train_iter:
            mask_CNN.train()
            C_XtoY.train()
            labels_X = labels_X.cuda()
            if iteration>opts.init_train_iter+opts.train_iters:break
            iteration+=1
            if (iteration-opts.init_train_iter) % 1000 == 0:
                opts.lr = opts.lr * 0.5
                g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
                print('lr',str(opts.lr))
                with open(opts.logfile,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'lr changed: '+str(opts.lr))
            if iteration == 10000 or iteration == 20000: 
                opts.w_image /= 2
                print('w_image',str(opts.w_image))
                with open(opts.logfile,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'w_image changed: '+str(opts.w_image))

            images_X = to_var(images_X)
            images_Y = to_var(torch.tensor(images_Y[0], dtype=torch.cfloat))

            # ============================================
            #            GENERATE TRAIING IMAGES
            # ============================================
            images_X_spectrum = [] # training images * opts.stack_imgs
            images_Y_spectrum = [] # training images * opts.stack_imgs
            for i in range(opts.stack_imgs):
                images_X_spectrum_raw = torch.stft(input=images_X.select(1,i), n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                                   win_length=opts.stft_window, pad_mode='constant');
                temp = spec_to_network_input(images_X_spectrum_raw, opts)
                temp1 = torch.angle(temp[:,0,:,:]+1j*temp[:,1,:,:])/np.pi
                temp2 = torch.abs(temp[:,0,:,:]+1j*temp[:,1,:,:])
                images_X_spectrum.append(torch.stack([temp1,temp2],1))
                
            
                images_Y_spectrum_raw = torch.stft(input=images_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                                   win_length=opts.stft_window, pad_mode='constant');
                temp = spec_to_network_input(images_Y_spectrum_raw, opts)
                temp = torch.abs(temp[:,0,:,:]+1j*temp[:,1,:,:])
                images_Y_spectrum.append(temp)
            
            #########################################
            ##              FORWARD
            #########################################
            if opts.model_ver == 0: fake_Y_spectrum = [mask_CNN(i) for i in images_X_spectrum]
            else: fake_Y_spectrum = mask_CNN(images_X_spectrum) #CNN input: a list of images, output: a list of images
            fake_Y_spectrum = torch.mean(torch.stack(fake_Y_spectrum,0),0) #average of CNN outputs
            
            g_y_pix_loss = loss_spec(fake_Y_spectrum, images_Y_spectrum[0])
            labels_X_estimated = C_XtoY(fake_Y_spectrum)
            g_y_class_loss = loss_class(labels_X_estimated, labels_X)
            g_optimizer.zero_grad()
            G_Image_loss = opts.w_image * g_y_pix_loss
            G_Class_loss = g_y_class_loss
            G_Y_loss = G_Image_loss + G_Class_loss 
            G_Y_loss.backward()
            g_optimizer.step()

            
            #########################################
            ##     LOG THE LOSS OF LATEST opts.log_step*5 STEPS
            #########################################
            if len(G_Y_loss_avg)<opts.log_step:
                G_Y_loss_avg.append(G_Y_loss.item())
                G_Image_loss_avg.append(G_Image_loss.item())
                G_Class_loss_avg.append(G_Class_loss.item())
            else:
                G_Y_loss_avg[iteration % opts.log_step] = G_Y_loss.item()
                G_Image_loss_avg[iteration % opts.log_step] = G_Image_loss.item()
                G_Class_loss_avg[iteration % opts.log_step] = G_Class_loss.item()
            if iteration % opts.log_step == 0:
                output_lst = ["{:6d}".format(iteration),
                                "{:6.3f}".format(np.mean(G_Y_loss_avg)),
                                "{:6.3f}".format(np.mean(G_Image_loss_avg)),
                                "{:6.3f}".format(np.mean(G_Class_loss_avg)),
                                "{:6.3f}".format(time.time() - oldtime)]
                output_str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' + ' '.join(output_lst)
                oldtime = time.time()
                print(output_str)
                with open(opts.logfile,'a') as f:
                    f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + ' , '.join(output_lst))

            ## checkpoint
            if (iteration) % opts.checkpoint_every == 0:
                checkpoint(iteration, mask_CNN, C_XtoY, opts)

            ## test
            if iteration % opts.test_step == 1:# or iteration == opts.init_train_iter + opts.train_iters:
                mask_CNN.eval()
                C_XtoY.eval()
                with torch.no_grad():
                    #print('start testing..')
                    error_matrix = 0
                    error_matrix_count = 0
                    test_iter = iter(testing_dataloader)
                    iteration2 = 0
                    G_Image_loss_avg_test = 0
                    G_Class_loss_avg_test = 0
                    for images_X_test, labels_X_test, images_Y_test0 in test_iter:
                            if iteration2 >= 100    : break
                            iteration2 += 1
                            labels_X_test = labels_X_test.cuda()

                            #prepare testing data
                            images_X_test, labels_X_test = to_var(images_X_test), to_var(labels_X_test)
                            images_Y_test = to_var(images_Y_test0[0])
                            images_X_test_spectrum = []
                            images_Y_test_spectrum = []
                            for i in range(opts.stack_imgs):
                                images_X_test_spectrum_raw = torch.stft(input=images_X_test.select(1,i), n_fft=opts.stft_nfft,
                                                                        hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                                        pad_mode='constant');
                                temp = spec_to_network_input(images_X_test_spectrum_raw, opts)
                                temp1 = torch.angle(temp[:,0,:,:]+1j*temp[:,1,:,:])/np.pi
                                temp2 = torch.abs(temp[:,0,:,:]+1j*temp[:,1,:,:])
                                images_X_test_spectrum.append(torch.stack([temp1,temp2],1))

                                images_Y_test_spectrum_raw = torch.stft(input=images_Y_test, n_fft=opts.stft_nfft,
                                                                        hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                                        pad_mode='constant');
                                temp = spec_to_network_input(images_Y_test_spectrum_raw, opts)
                                temp = torch.abs(temp[:,0,:,:]+1j*temp[:,1,:,:])
                                images_Y_test_spectrum.append(temp)

                            # forward
                            if opts.model_ver == 0: fake_Y_test_spectrums = [mask_CNN(i) for i in images_X_test_spectrum]
                            else: fake_Y_test_spectrums = mask_CNN(images_X_test_spectrum) #CNN input: a list of images, output: a list of images
                            fake_Y_test_spectrum = torch.mean(torch.stack(fake_Y_test_spectrums,0),0)
                            labels_X_estimated = C_XtoY(fake_Y_test_spectrum)

                            g_y_pix_loss = loss_spec(fake_Y_test_spectrum, images_Y_test_spectrum[0])
                            g_y_class_loss = loss_class(labels_X_estimated, labels_X_test)
                            G_Image_loss_avg_test += g_y_pix_loss.item() 
                            G_Class_loss_avg_test += g_y_class_loss.item() 

                            _, labels_X_test_estimated = torch.max(labels_X_estimated, 1)
                            test_right_case = (labels_X_test_estimated == labels_X_test)
                            error_matrix += np.sum(to_data(test_right_case))

                            error_matrix_count += opts.batch_size

                            if(iteration2==1):
                                save_samples(iteration+iteration2, images_Y_test_spectrum, images_X_test_spectrum, mask_CNN, opts)
                    error_matrix2 = error_matrix / error_matrix_count
                    print('TEST: ACC:' ,error_matrix2, '['+str(error_matrix)+'/'+str(error_matrix_count)+']','ILOSS:',"{:6.3f}".format(G_Image_loss_avg_test/error_matrix_count*opts.batch_size*opts.w_image) ,'CLOSS:',"{:6.3f}".format(G_Class_loss_avg_test/error_matrix_count*opts.batch_size))
                    with open(opts.logfile2,'a') as f:
                        f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    with open(opts.logfile,'a') as f:
                        f.write(' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    print('   CURRENT TIME       ITER  YLOSS  ILOSS  CLOSS  TIME  ----TRAINING----')
    return mask_CNN, C_XtoY
