# end2end.py
from __future__ import division
import os
import sys
from datetime import datetime
import torch.nn.functional as F

from scipy.signal import chirp, spectrogram
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
from utils import *
import torch.autograd.profiler as profiler
import time
import math
import scipy.io as scio


def checkpoint(iteration, models, opts):
    mask_CNN_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_maskCNN.pkl')
    create_dir(opts.checkpoint_dir)
    torch.save(models[0].state_dict(), mask_CNN_path)
    if opts.cxtoy == 'True':
        C_XtoY_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_C_XtoY.pkl')
        torch.save(models[1].state_dict(), C_XtoY_path)
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
    _, h, w = sources[0].shape
    row = 1# int(np.sqrt(opts.batch_size))
    column = 1# math.ceil(opts.batch_size / row)
    merged = np.zeros([row * h , column * w * opts.stack_imgs * 3])
    for stack_idx in range(opts.stack_imgs):
        for idx, (s, t, y) in enumerate(zip(sources[stack_idx], targets[stack_idx], Y[stack_idx] )):
            i = idx // column
            j = idx % column
            merged[i * h:(i + 1) * h,  (j * 3 * opts.stack_imgs + stack_idx*3)     * w:(j * 3 * opts.stack_imgs + 1 + stack_idx*3) * w,] = s
            merged[i * h:(i + 1) * h,  (j * 3 * opts.stack_imgs + stack_idx*3 + 1) * w:(j * 3 * opts.stack_imgs + 2 + stack_idx*3) * w,] = t
            merged[i * h:(i + 1) * h,  (j * 3 * opts.stack_imgs + stack_idx*3 + 2) * w:(j * 3 * opts.stack_imgs + 3 + stack_idx*3) * w,] = y
            break
    #merged = merged.transpose(1, 2, 0)
    #newsize = ( merged.shape[1] ,merged.shape[1] * opts.stack_imgs )
    #return cv2.resize(merged, dsize=newsize)
    return merged


def save_samples(iteration, fixed_Y, fixed_X, fake_Y, name, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fixed_X = [to_data(i) for i in fixed_X]
    Y, fake_Y = [to_data(i) for i in fixed_Y], [to_data(i) for i in fake_Y]

    mergeda = merge_images(fixed_X, fake_Y, Y, opts)

    path = os.path.join(opts.checkpoint_dir,
            'sample-{:06d}-snr{:.1f}-Y{:s}.png'.format(iteration,opts.snr_list[0],name))
    merged = np.abs(mergeda)
    merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged)) * 255
    merged = cv2.flip(merged, 0)
    #print(np.max(merged),np.min(merged),np.mean(merged))
    cv2.imwrite(path, merged)
    print('SAVED TEST SAMPLE: {}'.format(path))


def work2(fake_Y_spectrum, images_Y_spectrum,labels_X, C_XtoY, opts):
        g_y_pix_loss = opts.loss_spec(torch.abs(fake_Y_spectrum), torch.abs(images_Y_spectrum))

        G_Image_loss = opts.w_image * g_y_pix_loss
        if opts.cxtoy == 'True':
            labels_X_estimated = C_XtoY(fake_Y_spectrum)
        else:
            raise(NotImplementedError)
            fake_Y_test_spectrum2 = spec_to_network_input(torch.squeeze(fake_Y_spectrum), opts) # ??? 
            labels_X_estimated = F.softmax(fake_Y_test_spectrum2.sum(-1),dim=1)
        g_y_class_loss = opts.loss_class(labels_X_estimated, labels_X)
        G_Class_loss = g_y_class_loss
        _, labels_X_estimated = torch.max(labels_X_estimated, 1)
        test_right_case = to_data(labels_X_estimated == labels_X)
        G_Acc = np.sum(test_right_case) / opts.batch_size
        return G_Image_loss, G_Class_loss, G_Acc

def work(images_X, labels_X, images_Y, data_file_name, opts, downchirp, downchirpY, mask_CNN, C_XtoY):
    images_X = to_var(images_X)
    images_Y = to_var(torch.tensor(images_Y[0], dtype=torch.cfloat))
    if opts.dechirp == 'True': images_X = images_X * downchirp
    if opts.dechirp == 'True': images_Y = images_Y * downchirpY
    images_X_spectrum = []
    images_Y_spectrum = []
    for i in range(opts.stack_imgs):
        images_X_spectrum_raw = torch.stft(input=images_X.select(1,i), n_fft=opts.stft_nfft, 
                                            hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                            pad_mode='constant',return_complex=True)
        images_X_spectrum.append(spec_to_network_input(images_X_spectrum_raw, opts))
    
        images_Y_spectrum_raw = torch.stft(input=images_Y, n_fft=opts.stft_nfft, 
                                            hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                            pad_mode='constant',return_complex=True)
        images_Y_spectrum_raw = spec_to_network_input(images_Y_spectrum_raw, opts)
        images_Y_spectrum.append(images_Y_spectrum_raw)
    
    if opts.model_ver == 0:
        raise NotImplementedError
        fake_Y_spectrums = [mask_CNN(i) for i in images_X_spectrum]
    else:
        fake_Y_spectrums = mask_CNN(images_X_spectrum)

    if opts.avg_flag == 'True':
        fake_Y_spectrum_abs = torch.mean(torch.stack([torch.abs(fake_Y_spectrum) for fake_Y_spectrum in fake_Y_spectrums],0),0)

        fake_Y_spectrums = [fake_Y_spectrum * (fake_Y_spectrum_abs / torch.abs(fake_Y_spectrum)) for fake_Y_spectrum in fake_Y_spectrums]

    G_Y_loss = 0
    G_Image_loss = 0
    G_Class_loss = 0
    G_Acc = 0
    for i in range(opts.stack_imgs):
        G_Image_loss_img, G_Class_loss_img, G_Acc_img = work2(fake_Y_spectrums[i], images_Y_spectrum[i],labels_X,C_XtoY, opts)
        G_Image_loss += G_Image_loss_img / opts.stack_imgs
        G_Class_loss += G_Class_loss_img / opts.stack_imgs
        G_Acc += G_Acc_img / opts.stack_imgs
    if opts.iteration % opts.test_step == 1: 
        save_samples(opts.iteration, images_Y_spectrum, images_X_spectrum, fake_Y_spectrums, 'val', opts)
    return G_Image_loss, G_Class_loss, G_Acc


def training_loop(training_dataloader,testing_dataloader, models, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
    """
    mask_CNN = models[0]
    if opts.cxtoy == 'True': C_XtoY = models[1] 
    else: C_XtoY = None
    opts.loss_spec = torch.nn.MSELoss(reduction='mean')
    opts.loss_class = nn.CrossEntropyLoss()
    opts.iteration = 0


    if opts.dechirp == 'True':
        nsamp = int(opts.fs * opts.n_classes / opts.bw)
        t = np.linspace(0, nsamp / opts.fs, nsamp)
        chirpI1 = chirp(t, f0=opts.bw/2, f1=-opts.bw/2, t1=2** opts.sf / opts.bw , method='linear', phi=90)
        chirpQ1 = chirp(t, f0=opts.bw/2, f1=-opts.bw/2, t1=2** opts.sf / opts.bw, method='linear', phi=0)
        dechirp = chirpI1+1j*chirpQ1
        downchirp1 = torch.tensor(dechirp, dtype=torch.cfloat).cuda()
        downchirp = torch.stack([ torch.stack([ downchirp1 for i in range(opts.stack_imgs)])for i in range(opts.batch_size)])
        downchirpY = torch.stack([ downchirp1 for i in range(opts.batch_size)])


    
    g_params = list(mask_CNN.parameters()) 
    if opts.cxtoy == 'True': g_params += list(C_XtoY.parameters())
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    G_Y_loss_avg = 0
    G_Image_loss_avg = 0
    G_Class_loss_avg = 0
    G_Acc_avg = 0

    iteration = opts.init_train_iter
    oldtime = time.time()
    scoreboards = [0, 0,]
    trim_size = opts.freq_size // 2
    print('   CURRENT TIME       ITER  YLOSS  ILOSS  CLOSS   ACC   TIME  ----TRAINING',opts.lr,'----')
    while iteration<=opts.init_train_iter+opts.train_iters:
            iteration+=1
            mask_CNN.train()
            if opts.cxtoy == 'True':C_XtoY.train()

            images_X, labels_X, images_Y, data_file_name = next(training_dataloader.__iter__())
            g_optimizer.zero_grad()
            G_Image_loss, G_Class_loss, G_Acc =  work(images_X, labels_X, images_Y, data_file_name, opts, downchirp, downchirpY, mask_CNN, C_XtoY)

            G_Y_loss = G_Image_loss + G_Class_loss 
            G_Y_loss.backward()
            g_optimizer.step()

            G_Y_loss_avg += G_Y_loss.item()
            G_Image_loss_avg += G_Image_loss.item()
            G_Class_loss_avg += G_Class_loss.item()
            G_Acc_avg += G_Acc

            if iteration % opts.log_step == 0:
                output_lst = [  "{:6d}".format(iteration),
                                "{:6.3f}".format(G_Y_loss_avg / opts.log_step ),
                                "{:6.3f}".format(G_Image_loss_avg / opts.log_step),
                                "{:6.3f}".format(G_Class_loss_avg / opts.log_step),
                                "{:6.3f}".format(G_Acc_avg / opts.log_step),
                                "{:6.3f}".format(time.time() - oldtime)]
                output_str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' + ' '.join(output_lst)
                oldtime = time.time()
                print(output_str)
                with open(opts.logfile,'a') as f: f.write('\n'+output_str)
                G_Y_loss_avg = 0
                G_Image_loss_avg = 0
                G_Class_loss_avg = 0
                G_Acc_avg = 0
            if (iteration) % opts.checkpoint_every == 0: checkpoint(iteration, models, opts)

            if iteration % opts.test_step == 1:# or iteration == opts.init_train_iter + opts.train_iters:
                mask_CNN.eval()
                if opts.cxtoy == 'True':C_XtoY.eval()
                with torch.no_grad():
                    #print('start testing..')
                    error_matrix = 0
                    error_matrix_count = 0
                    iteration2 = 0
                    G_Image_loss_avg_test = 0
                    G_Class_loss_avg_test = 0
                    while iteration2 < opts.max_test_iters: 
                            opts.iteration = iteration + iteration2
                            iteration2 += 1
                            images_X_test, labels_X_test, images_Y_test, data_file_name = next(testing_dataloader.__iter__())
                            G_Image_loss, G_Class_loss, G_Acc = work(images_X_test, labels_X_test, images_Y_test, data_file_name, opts, downchirp, downchirpY, mask_CNN, C_XtoY)
                            G_Image_loss_avg_test += G_Image_loss
                            G_Class_loss_avg_test += G_Class_loss

                            error_matrix += G_Acc
                            error_matrix_count += 1

                    error_matrix2 = error_matrix / error_matrix_count
                    print('TEST: ACC:' ,error_matrix2, '['+str(error_matrix)+'/'+str(error_matrix_count)+']','ILOSS:',"{:6.3f}".format(G_Image_loss_avg_test/error_matrix_count) ,
                            'CLOSS:',"{:6.3f}".format(G_Class_loss_avg_test/error_matrix_count))
                    with open(opts.logfile2,'a') as f:
                        f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    with open(opts.logfile,'a') as f:
                        f.write(' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    if(error_matrix2>=opts.terminate_acc):
                        print('REACHED',opts.terminate_acc,'ACC, TERMINATINg...')
                        iteration = opts.init_train_iter + opts.train_iters + 1
                        break
                    print('   CURRENT TIME       ITER  YLOSS  ILOSS  CLOSS   ACC   TIME  ----TRAINING',opts.lr,'----')
    return [mask_CNN, C_XtoY]
