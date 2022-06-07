# end2end.py
from __future__ import division
import os
import sys
from datetime import datetime
import torch.nn.functional as F

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

def merge_images(sources, targets, Y, test_right_case, opts):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources[0].shape
    row = int(np.sqrt(opts.batch_size))
    column = math.ceil(opts.batch_size / row)
    merged = np.zeros([opts.y_image_channel, row * h * opts.stack_imgs, column * w * 3])
    for stack_idx in range(opts.stack_imgs):
        for idx, (s, t, y, c) in enumerate(zip(sources[stack_idx], targets[stack_idx], Y[stack_idx], test_right_case)):
            i = idx // column
            j = idx % column
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3) * w:(j * 3 + 1) * w] = s
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3 + 1) * w:(j * 3 + 2) * w] = t
            merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3 + 2) * w:(j * 3 + 3) * w] = y
            if not c:
                 merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3) * w:(j * 3) * w + 1] = np.max(merged)*0.7
                 merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3 + 1) * w - 1:(j * 3 + 1) * w] = np.max(merged)*0.7
                 merged[:, (i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + stack_idx) * h + 1, (j * 3) * w:(j * 3 + 1) * w] = np.max(merged)*0.7
                 merged[:, (i * opts.stack_imgs + 1 + stack_idx) * h - 1:(i * opts.stack_imgs + 1 + stack_idx) * h, (j * 3) * w:(j * 3 + 1) * w] = np.max(merged)*0.7
    merged = merged.transpose(1, 2, 0)
    print(merged.shape)
    newsize = ( merged.shape[1] ,merged.shape[1] * opts.stack_imgs )
    print(newsize)
    #return cv2.resize(merged, dsize=newsize)
    return merged


def save_samples(iteration, fixed_Y, fixed_X, mask_CNN, test_right_case, name, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    if opts.model_ver == 0: fake_Y = mask_CNN(torch.cat(fixed_X,1))
    else: fake_Y = mask_CNN(fixed_X)
    fixed_X = [to_data(i) for i in fixed_X]

    Y, fake_Y = [to_data(i) for i in fixed_Y], [to_data(i) for i in fake_Y]

    mergeda = merge_images(fixed_X, fake_Y, Y, test_right_case, opts)

    path = os.path.join(opts.checkpoint_dir,
            'sample-{:06d}{:d}-Y{:s}.png'.format(iteration,opts.snr_list[0],name))
    merged = np.abs(mergeda[:, :, 0]+1j*mergeda[:, :, 1])
    merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged)) * 255
    merged = cv2.flip(merged, 0)
    #print(np.max(merged),np.min(merged),np.mean(merged))
    cv2.imwrite(path, merged)
    print('SAMPLE: {}'.format(path))


def training_loop(training_dataloader, val_dataloader, testing_dataloader,mask_CNN, C_XtoY, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
    """
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = nn.CrossEntropyLoss()


    # load downchirp

    if opts.dechirp == 'True':
        assert opts.data_format == 2
        fname = '0_35_'+str(opts.sf)+'_'+str(opts.bw)+'_0_0_1_1.mat'
        path = os.path.join(opts.data_dir,fname)
        path = path.replace('test','new')
        print('LOADING DECHIRP FROM',path)
        lora_img = np.array(scio.loadmat(path)[opts.feature_name].tolist())
        lora_img = np.squeeze(lora_img)
        lora_img = lora_img[::-1].copy()

        downchirp1 = torch.tensor(lora_img, dtype=torch.cfloat).cuda()
        downchirp0 = torch.conj(downchirp1).clone()
        downchirp = torch.stack([ downchirp0 for i in range(opts.stack_imgs)])
        downchirp = torch.stack([ downchirp for i in range(opts.batch_size)])
        downchirpY = torch.stack([ downchirp0 for i in range(opts.batch_size)])


    
    # Create generators and discriminators
    g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])

    G_Y_loss_avg = []
    G_Image_loss_avg = []
    G_Class_loss_avg = []
    G_Acc = 0

    iteration = opts.init_train_iter
    oldtime = time.time()#time the training process

    scoreboards = [0, 0,]
    linefilter = torch.tensor([[opts.w_line / (abs(i-k)  + opts.w_line ) for i in range(opts.n_classes)] for k in range(opts.n_classes)])
    linefilter = linefilter.unsqueeze(2).repeat(1, 1, 33) ##debug images_Y_spectrum_raw=33

    trim_size = opts.freq_size // 2
    linefilter = torch.cat((linefilter[:, -trim_size:, :], linefilter[:, 0:trim_size, :]), 1)
    linefilter = linefilter.cuda()

    while iteration < opts.init_train_iter + opts.train_iters:
        train_iter = iter(training_dataloader)
        if iteration !=  opts.init_train_iter:  print('============WARNING: REUSING DATA, STARTING NEW TRAINING EPOCH=============')
        for images_X, labels_X, images_Y, data_file_name in train_iter:
            linefilter_X = linefilter[labels_X,:,:]
            mask_CNN.train()
            C_XtoY.train()
            labels_X = labels_X.cuda()
            if iteration>opts.init_train_iter+opts.train_iters:break
            iteration+=1
            if iteration == 20000 or iteration == 40000: 
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
            if opts.dechirp == 'True': images_X = images_X * downchirp
            if opts.dechirp == 'True': images_Y = images_Y * downchirpY
            for i in range(opts.stack_imgs):
                images_X_spectrum_raw = torch.stft(input=images_X.select(1,i), n_fft=opts.stft_nfft, 
                                                    hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                                    pad_mode='constant',return_complex=True)
                images_X_spectrum.append(spec_to_network_input2( spec_to_network_input(images_X_spectrum_raw, opts), opts ))
            
                images_Y_spectrum_raw = torch.stft(input=images_Y, n_fft=opts.stft_nfft, 
                                                    hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                                    pad_mode='constant',return_complex=True)
                images_Y_spectrum_raw = spec_to_network_input(images_Y_spectrum_raw, opts)
                if opts.line == 'True': 
                    images_Y_spectrum_raw *= linefilter_X
                images_Y_spectrum_raw = spec_to_network_input2(images_Y_spectrum_raw, opts)
                images_Y_spectrum.append(images_Y_spectrum_raw)

            
            #########################################
            ##              FORWARD
            #########################################
            g_optimizer.zero_grad()
            G_Y_loss = 0
            G_Image_loss = 0
            G_Class_loss = 0
            if opts.model_ver == 0:
                fake_Y_spectrum = mask_CNN(torch.cat(images_X_spectrum,1))
                for i in range(opts.stack_imgs):
                    g_y_pix_loss = loss_spec(fake_Y_spectrum[:,i*2:i*2+2,:,:], images_Y_spectrum[i])
                    labels_X_estimated = C_XtoY(fake_Y_spectrum[:,i*2:i*2+2,:,:])
                    g_y_class_loss = loss_class(labels_X_estimated, labels_X)
                    G_Image_loss = opts.w_image * g_y_pix_loss
                    G_Class_loss = g_y_class_loss
                    G_Y_loss += opts.w_image * g_y_pix_loss + g_y_class_loss 
                    _, labels_X_estimated = torch.max(labels_X_estimated, 1)
                    test_right_case = to_data(labels_X_estimated == labels_X)
                    G_Acc += np.sum(test_right_case)/opts.stack_imgs
            else: 
                #with torch.no_grad():
                fake_Y_spectrums = mask_CNN(images_X_spectrum) #CNN input: a list of images, output: a list of images
                if opts.cxtoy_each == 'True':
                    for i in range(opts.stack_imgs):
                        fake_Y_spectrum = fake_Y_spectrums[i]
                        g_y_pix_loss1 = loss_spec(fake_Y_spectrum, images_Y_spectrum[0])
                        g_y_pix_loss2 = loss_spec(torch.abs(fake_Y_spectrum[:,0]+1j*fake_Y_spectrum[:,1]), torch.abs(images_Y_spectrum[0][:,0]+1j*images_Y_spectrum[0][:,1]))
                        if opts.image_loss_abs=='True': g_y_pix_loss = g_y_pix_loss1
                        elif opts.image_loss_abs=='False': g_y_pix_loss = g_y_pix_loss2
                        else: raise NotImplementedError

                        G_Image_loss += opts.w_image * g_y_pix_loss / opts.stack_imgs
                        if opts.cxtoy == 'True':
                            labels_X_estimated = C_XtoY(fake_Y_spectrum)
                        else:
                            fake_Y_test_spectrum2 = spec_to_network_input(torch.abs(fake_Y_spectrum[:,0,:,:] + 1j*fake_Y_spectrum[:,1,:,:]), opts)
                            labels_X_estimated = F.softmax(fake_Y_test_spectrum2.sum(-1),dim=1)
                        g_y_class_loss = loss_class(labels_X_estimated, labels_X)
                        G_Class_loss += g_y_class_loss / opts.stack_imgs
                        _, labels_X_estimated = torch.max(labels_X_estimated, 1)
                        test_right_case = to_data(labels_X_estimated == labels_X)
                        G_Acc += np.sum(test_right_case)
                elif opts.cxtoy_each == 'False':
                    fake_Y_spectrum = torch.mean(torch.stack(fake_Y_spectrums,0),0) #average of CNN outputs
                    g_y_pix_loss1 = loss_spec(fake_Y_spectrum, images_Y_spectrum[0])
                    g_y_pix_loss2 = loss_spec(torch.abs(fake_Y_spectrum[:,0]+1j*fake_Y_spectrum[:,1]), torch.abs(images_Y_spectrum[0][:,0]+1j*images_Y_spectrum[0][:,1]))
                    if opts.image_loss_abs=='True': g_y_pix_loss = g_y_pix_loss1
                    elif opts.image_loss_abs=='False': g_y_pix_loss = g_y_pix_loss2
                    else: raise NotImplementedError

                    G_Image_loss = opts.w_image * g_y_pix_loss
                    if opts.cxtoy == 'True':
                        labels_X_estimated = C_XtoY(fake_Y_spectrum)
                        g_y_class_loss = loss_class(labels_X_estimated, labels_X)
                        G_Class_loss = g_y_class_loss
                        _, labels_X_estimated = torch.max(labels_X_estimated, 1)
                        test_right_case = to_data(labels_X_estimated == labels_X)
                        G_Acc += np.sum(test_right_case)*opts.stack_imgs
                    else: raise NotImplementedError
                else: raise NotImplementedError

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
                #if opts.cxtoy == 'True': G_Class_loss_avg.append(G_Class_loss.item())
                #else: G_Class_loss_avg.append(0)
            else:
                G_Y_loss_avg[iteration % opts.log_step] = G_Y_loss.item()
                G_Image_loss_avg[iteration % opts.log_step] = G_Image_loss.item()
                G_Class_loss_avg[iteration % opts.log_step] = G_Class_loss.item()
                #if opts.cxtoy == 'True': G_Class_loss_avg[iteration % opts.log_step] = G_Class_loss.item()
                #else: G_Class_loss_avg[iteration % opts.log_step] = 0
            if iteration % opts.log_step == 0:
                output_lst = ["{:6d}".format(iteration),
                                "{:6.3f}".format(np.mean(G_Y_loss_avg)),
                                "{:6.3f}".format(np.mean(G_Image_loss_avg)),
                                "{:6.3f}".format(np.mean(G_Class_loss_avg)),
                                "{:6.3f}".format(G_Acc / opts.log_step / opts.batch_size / opts.stack_imgs),
                                "{:6.3f}".format(time.time() - oldtime)]
                output_str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' + ' '.join(output_lst)
                oldtime = time.time()
                print(output_str)
                with open(opts.logfile,'a') as f:
                    f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + ' , '.join(output_lst))
                G_Acc = 0

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
                    val_iter = iter(val_dataloader)
                    iteration2 = 0
                    G_Image_loss_avg_test = 0
                    G_Class_loss_avg_test = 0
                    sample_cnt = 0
                    for images_X_test, labels_X_test, images_Y_test0, data_file_name in val_iter:
                            linefilter_X = linefilter[labels_X_test,:,:]
                            if iteration2 >= 10: break
                            iteration2 += 1
                            labels_X_test = labels_X_test.cuda()

                            #prepare testing data
                            images_X_test, labels_X_test = to_var(images_X_test), to_var(labels_X_test)
                            if(opts.flank>=0 and iteration2 < 5):
                                print('======WARNING: FLANKING IMAGES TO', opts.flank, '=========')
                            #for i in range(opts.stack_imgs):
                            for i in range(opts.flank):
                                images_X_test[:,i,:] = images_X_test[:,opts.flank,:]
                            images_Y_test = to_var(images_Y_test0[0])
                            images_X_test_spectrum = []
                            images_Y_test_spectrum = []
                            if opts.dechirp == 'True': images_X_test = images_X_test* downchirp
                            if opts.dechirp == 'True': images_Y_test = images_Y_test * downchirpY
                            for i in range(opts.stack_imgs):
                                images_X_test_spectrum_raw = torch.stft(input=images_X_test.select(1,i), n_fft=opts.stft_nfft, #(opts.n_classes* opts.fs // opts.bw)
                                                    hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                                    pad_mode='constant',return_complex=True)
                                images_X_test_spectrum.append(spec_to_network_input2(spec_to_network_input(images_X_test_spectrum_raw, opts),opts))

                                images_Y_test_spectrum_raw = torch.stft(input=images_Y_test, n_fft=opts.stft_nfft,
                                                    hop_length=opts.stft_overlap , win_length=opts.stft_window,
                                                    pad_mode='constant',return_complex=True)
                                images_Y_test_spectrum_raw = spec_to_network_input(images_Y_test_spectrum_raw, opts)
                                if opts.line == 'True': 
                                    images_Y_test_spectrum_raw *= linefilter_X
                                images_Y_test_spectrum_raw = spec_to_network_input2(images_Y_test_spectrum_raw, opts)
                                images_Y_test_spectrum.append(images_Y_test_spectrum_raw)

                            # forward
                            if opts.model_ver == 0: 
                                fake_Y_test_spectrum = mask_CNN(torch.cat(images_X_test_spectrum, 1))
                                labels_X_estimated = C_XtoY(fake_Y_test_spectrum[:,:2,:,:])
                                g_y_pix_loss = loss_spec(fake_Y_test_spectrum[:,:2,:,:], images_Y_test_spectrum[0])
                            else: 
                                fake_Y_test_spectrums = mask_CNN(images_X_test_spectrum) #CNN input: a list of images, output: a list of images
                                fake_Y_test_spectrum = torch.mean(torch.stack(fake_Y_test_spectrums,0),0)
                                g_y_pix_loss = loss_spec(fake_Y_test_spectrum, images_Y_test_spectrum[0])
                                if opts.cxtoy=='True':
                                    labels_X_estimated = C_XtoY(fake_Y_test_spectrum)
                                else:
                                    fake_Y_test_spectrums = [spec_to_network_input(torch.abs(x[:,0,:,:] + 1j*x[:,1,:,:]), opts) for x in fake_Y_test_spectrums]
                                    fake_Y_test_spectrum3 = torch.stack(fake_Y_test_spectrums,0)
                                    labels_X_estimated =  F.softmax(fake_Y_test_spectrum3.sum(0).sum(-1),dim=1)#.reshape(opts.batch_size, opts.n_classes)


                            g_y_class_loss = loss_class(labels_X_estimated, labels_X_test)
                            G_Image_loss_avg_test += g_y_pix_loss.item() 
                            G_Class_loss_avg_test += g_y_class_loss.item() 

                            _, labels_X_test_estimated = torch.max(labels_X_estimated, 1)
                            test_right_case = to_data(labels_X_test_estimated == labels_X_test)
                            error_matrix += np.sum(test_right_case)

                            error_matrix_count += opts.batch_size

                            if(sample_cnt==0 and  np.sum(test_right_case) < opts.batch_size  ):
                                sample_cnt+=1
                                print(labels_X_test_estimated, labels_X_test,test_right_case.astype(np.int))
                                #print(data_file_name)
                                save_samples(iteration+iteration2, images_Y_test_spectrum, images_X_test_spectrum, mask_CNN, test_right_case, 'val', opts)
                    error_matrix2 = error_matrix / error_matrix_count
                    print('TEST: ACC:' ,error_matrix2, '['+str(error_matrix)+'/'+str(error_matrix_count)+']','ILOSS:',"{:6.3f}".format(G_Image_loss_avg_test/error_matrix_count*opts.batch_size*opts.w_image) ,'CLOSS:',"{:6.3f}".format(G_Class_loss_avg_test/error_matrix_count*opts.batch_size))
                    with open(opts.logfile2,'a') as f:
                        f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    with open(opts.logfile,'a') as f:
                        f.write(' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
                    if error_matrix2 <= scoreboards[-1] and error_matrix2 <= scoreboards[-2] and opts.lr>=0.00001 and iteration - opts.init_train_iter >= opts.start_lr_decay:
                        opts.lr = opts.lr * 0.5
                        g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
                        print('------------INSUFFICIENT PROGRESS, DOWNGRADING LREANING RATE TO',str(opts.lr),'------------')
                        with open(opts.logfile,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'lr changed: '+str(opts.lr))
                        scoreboards = [0, 0]
                    else:
                        scoreboards.append(error_matrix2)
                    if(error_matrix2>=opts.terminate_acc):
                        print('REACHED',opts.terminate_acc,'ACC, TERMINATINg...')
                        iteration = opts.init_train_iter + opts.train_iters + 1
                        break
                    print('   CURRENT TIME       ITER  YLOSS  ILOSS  CLOSS   ACC   TIME  ----TRAINING',opts.lr,'----')
    return mask_CNN, C_XtoY
    '''
except KeyboardInterrupt:
    print(' KEYBOARD INTERRUPT, PERFORMING FINAL TEST BEFORE EXITING...')
except Exception as e:
    raise e
finally:
    print('-------------------- FINAL TEST --------------------')
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
        sample_cnt = 0
        for images_X_test, labels_X_test, images_Y_test0, data_file_name in test_iter:
                #if iteration2 >= opts.max_test_iters: break
                iteration2 += 1
                labels_X_test = labels_X_test.cuda()

                #prepare testing data
                images_X_test, labels_X_test = to_var(images_X_test), to_var(labels_X_test)
                if(opts.flank>=0 and iteration2 < 5):
                    print('======WARNING: FLANKING IMAGES TO', opts.flank, '=========')
                #for i in range(opts.stack_imgs):
                for i in range(opts.flank):
                    images_X_test[:,i,:] = images_X_test[:,opts.flank,:]
                images_Y_test = to_var(images_Y_test0[0])
                images_X_test_spectrum = []
                images_Y_test_spectrum = []
                for i in range(opts.stack_imgs):
                    images_X_test_spectrum_raw = torch.stft(input=images_X_test.select(1,i), n_fft=opts.stft_nfft,
                                        hop_length=opts.stft_overlap , win_length=opts.stft_window ,
                                        pad_mode='constant',return_complex=True)
                    images_X_test_spectrum.append(spec_to_network_input(images_X_test_spectrum_raw, opts))

                    images_Y_test_spectrum_raw = torch.stft(input=images_Y_test, n_fft=opts.stft_nfft,
                                        hop_length=opts.stft_overlap , win_length=opts.stft_window,
                                        pad_mode='constant',return_complex=True)
                    images_Y_test_spectrum.append(spec_to_network_input(images_Y_test_spectrum_raw, opts))

                # forward
                if opts.model_ver == 0: 
                    fake_Y_test_spectrum = mask_CNN(torch.cat(images_X_test_spectrum, 1))
                    labels_X_estimated = C_XtoY(fake_Y_test_spectrum[:,:2,:,:])
                    g_y_pix_loss = loss_spec(fake_Y_test_spectrum[:,:2,:,:], images_Y_test_spectrum[0])
                else: 
                    fake_Y_test_spectrums = mask_CNN(images_X_test_spectrum) #CNN input: a list of images, output: a list of images
                    fake_Y_test_spectrum = torch.mean(torch.stack(fake_Y_test_spectrums,0),0)
                    labels_X_estimated = C_XtoY(fake_Y_test_spectrum)
                    g_y_pix_loss = loss_spec(fake_Y_test_spectrum, images_Y_test_spectrum[0])

                g_y_class_loss = loss_class(labels_X_estimated, labels_X_test)
                G_Image_loss_avg_test += g_y_pix_loss.item() 
                G_Class_loss_avg_test += g_y_class_loss.item() 

                _, labels_X_test_estimated = torch.max(labels_X_estimated, 1)
                test_right_case = to_data(labels_X_test_estimated == labels_X_test)
                error_matrix += np.sum(test_right_case)

                error_matrix_count += opts.batch_size

                if(sample_cnt==0 and  np.sum(test_right_case) < opts.batch_size  ):
                    sample_cnt+=1
                    print(labels_X_test_estimated, labels_X_test,test_right_case.astype(np.int))
                    #print(data_file_name)
                    save_samples(iteration+iteration2, images_Y_test_spectrum, images_X_test_spectrum, mask_CNN, test_right_case, 'test', opts)
        error_matrix2 = error_matrix / error_matrix_count
        print('TEST: ACC:' ,error_matrix2, '['+str(error_matrix)+'/'+str(error_matrix_count)+']','ILOSS:',"{:6.3f}".format(G_Image_loss_avg_test/error_matrix_count*opts.batch_size*opts.w_image) ,'CLOSS:',"{:6.3f}".format(G_Class_loss_avg_test/error_matrix_count*opts.batch_size))
        with open(opts.logfile2,'a') as f:
            f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
        with open(opts.logfile,'a') as f:
            f.write(' , ' + "{:6d}".format(iteration) +  ' , ' + "{:6.3f}".format(error_matrix2))
    return mask_CNN, C_XtoY'''
