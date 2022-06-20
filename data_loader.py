import os
import random
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import pickle
import sys
from utils import to_var, to_data, spec_to_network_input, create_dir
import cv2
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import math

class lora_dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, opts ):
        self.opts = opts
        self.initFlag = 0

    def __len__(self):
        return np.iinfo(np.int64).max

    def __getitem__(self, index0):
            try:
                data_perY = []
                symbol_index = random.randint(0,self.opts.n_classes-1)
                for k in range(self.opts.stack_imgs):
                    nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw)
                    t = np.linspace(0, nsamp / self.opts.fs, nsamp)
                    phi = random.uniform(-90, 90)
                    chirpI = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=phi+90)
                    chirpQ = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=phi)
                    mchirp0 = chirpI+1j*chirpQ
                    mchirp = np.tile(mchirp0, 2)

                    '''
                    for symbol_index in range(256):
                        time_shift = round((self.opts.n_classes - symbol_index) / self.opts.n_classes * nsamp)
                        chirp_raw = mchirp[time_shift:time_shift+nsamp]

                        chirpI1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=90)
                        chirpQ1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=0)
                        dechirp = chirpI1+1j*chirpQ1

                        print(time_shift, nsamp, mchirp.shape)
                        data_perY[0] = torch.tensor(chirp_raw*1e-6)

                        images_X_SpF_spectrum_raw = torch.stft(input= torch.tensor(chirp_raw * dechirp), n_fft=self.opts.stft_nfft, 
                                        hop_length=self.opts.stft_overlap // self.opts.stack_imgs, win_length=self.opts.stft_window // self.opts.stack_imgs, 
                                        pad_mode='constant') 
                        freq_size = self.opts.freq_size 
                        # trim 
                        trim_size = freq_size // 2 
                        # up down 拼接 
                        # images_X_SpF_spectrum_raw = torch.cat((images_X_SpF_spectrum_raw[-trim_size:, :], images_X_SpF_spectrum_raw[0:trim_size, :]), 0) 
                        print(images_X_SpF_spectrum_raw.shape,'images_X_SpF_spectrum_raw') 


                        merged = to_data(np.abs(images_X_SpF_spectrum_raw)) 
                        merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged)) * 255 
                        merged = np.squeeze(merged) 
                        merged = cv2.flip(merged, 0) 
                        cv2.imwrite('SpFData'+str(symbol_index)+'.png', merged) 
                    sys.exit(1)'''

                    time_shift = round((self.opts.n_classes - symbol_index) / self.opts.n_classes * nsamp)
                    chirp_raw = mchirp[time_shift:time_shift+nsamp]
                    data_perY.append( torch.tensor(chirp_raw, dtype = torch.cfloat).cuda())

                data_pers = []
                for k in range(self.opts.stack_imgs):
                        snr = self.opts.snr_list[k]
                        amp = math.pow(0.1, snr/20)
                        noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat).cuda()
                        data = data_perY[k]
                        data = data_perY[k] + noise

                        data_pers.append(data)
                        #data_file_name = [(self.opts.n_classes - symbol_index)%self.opts.n_classes, snr, self.opts.sf, self.opts.bw, 1, symbol_index, 1, 1]
                        #data_file_name = '_'.join([str(i) for i in data_file_name])+'.mat'
                label_per = (self.opts.n_classes - symbol_index)%self.opts.n_classes
                label_per = torch.tensor(label_per, dtype=int).cuda()
                data_file_name = [(self.opts.n_classes - symbol_index)%self.opts.n_classes, 35, self.opts.sf, self.opts.bw, 1, symbol_index, 1, 1]
                data_file_name = '_'.join([str(i) for i in data_file_name])+'.mat'

                data_pers = torch.stack(data_pers).cuda()

                return data_pers, label_per, data_perY, data_file_name
            except ValueError as e:
                print(e, self.data_lists[index % len(self.data_lists)])
                raise e
            except OSError as e:
                print(e, self.data_lists[index % len(self.data_lists)])
                raise e



# receive the csi feature map derived by the ray model as the input
def lora_loader(opts):
    training_dataset = lora_dataset(opts )

    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True)
    return training_dloader



