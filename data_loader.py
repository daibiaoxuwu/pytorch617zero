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

    def __init__(self, opts, files_list):
        self.opts = opts
        self.data_lists = files_list
        if opts.data_format == 3:
            self.folders_list = next(os.walk(opts.data_dir))[1][:opts.stack_imgs]
        self.initFlag = 0

    def __len__(self):
        return len(self.data_lists) 

    def load_img(self, path):
        lora_img = np.array(scio.loadmat(path)[self.opts.feature_name].tolist())
        lora_img = np.squeeze(lora_img)
        return torch.tensor(lora_img, dtype=torch.cfloat)


    def __getitem__(self, index0):
        for index in range(index0, index0 + len(self.data_lists)):
            try:
                data_file_name = self.data_lists[index % len(self.data_lists)]
                data_file_names = [data_file_name,]
                data_file_parts = data_file_name.split('_')
                
                label_per = int(data_file_parts[0].split('/')[-1])
                label_per = torch.tensor(label_per, dtype=int).cuda()

                if self.opts.data_format == 3:
                    paths = [os.path.join(self.opts.data_dir, folder, data_file_name) for folder in self.folders_list]
                    data_perY = [self.load_img(path).cuda() for path in paths]
                elif self.opts.data_format == 2: #DEBUG!!! LOAD -15 INSTEAD OF 35 FOR FORMAT==2
                    data_perY = []
                    for k in range(self.opts.stack_imgs):
                        nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw)
                        t = np.linspace(0, nsamp / self.opts.fs, nsamp)
                        phi = random.randint(-90, 90)
                        chirpI = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=phi+90)
                        chirpQ = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=phi)
                        mchirp0 = chirpI+1j*chirpQ
                        mchirp = np.tile(mchirp0, 2)
                        symbol_index =   int(data_file_parts[5])
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
                        if snr == -15:
                            amp = 5.6234
                        elif snr == -20:
                            amp = 10
                        elif snr == -25:
                            amp = 17.7828
                        else:
                            raise NotImplementedError
                        noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat).cuda()
                        data = data_perY[k]
                        data = data_perY[k] + noise

                        data_pers.append(data)

                ### ABOUT SPF

                data_pers = torch.stack(data_pers).cuda()

                return data_pers, label_per, data_perY, data_file_names
            except ValueError as e:
                print(e, self.data_lists[index % len(self.data_lists)])
                raise e
            except OSError as e:
                print(e, self.data_lists[index % len(self.data_lists)])
                raise e

        raise StopIteration 


# receive the csi feature map derived by the ray model as the input
def lora_loader(opts):
    if opts.data_format == 1:

        files = os.listdir(os.path.join(opts.data_dir))
        files_train = list(filter(lambda i: i.split('_')[1] == '35' and int(i.split('_')[4]) < 90 and i.split('_')[7].split('.')[0] == '1', files))
        files_test = list(filter(lambda i: i.split('_')[1] == '35' and int(i.split('_')[4]) >= 90 and i.split('_')[7].split('.')[0] == '1', files))
        random.shuffle(files_train)
        random.shuffle(files_test)
        opts.feature_name = 'chirp_new_SpF'
    elif opts.data_format == 2:
        files = os.listdir(os.path.join(opts.data_dir))

        snr_all = set( [int(i.split('_')[1]) for i in files])
        main_snr = opts.snr_list[0]
        files_35 = list(filter(lambda i: i.split('_')[1] == str(35) ,files))
        vals_35 = set([i.split('_')[0] for i in os.listdir('/data/djl/sf'+str(opts.sf)+'_125k_new')]) 
        files_all = set(filter(lambda i: i.split('_')[1] == str(main_snr) and i.split('_')[0] in vals_35 ,files))
        files_set = set(files)
        for snr in snr_all:
            if snr==35:continue
            if snr==main_snr:continue
            if snr not in opts.snr_list: continue
            files_failed = set()
            for filename in files_all:
                filename_parts = filename.split('_')
                filename_parts[1] = str(snr)
                filename_new = '_'.join(filename_parts)
                if filename_new not in files_set: files_failed.add(filename)
            files_all -= files_failed
                
        templ = list(set([int(i.split('_')[4]) for i in files_all]))
        #templ.sort()
        #for k in templ: print(k,len(set(filter(lambda i: int(i.split('_')[4]) == k,files))))
        for i in opts.snr_list: 
            if i not in snr_all: 
                print('SNR LIST FAILURE',opts.snr_list, i, snr_all)
                sys.exit(1)

        if max(templ)<50:
            files_train = list(files_all)
            files_test = list(files_all)
            files_val = list(files_all)
        else:
            split = int(max(templ) - 8)
            split2 = int(max(templ) - 4)
            files_train = list(filter(lambda i: int(i.split('_')[4]) < split,files_all))
            files_val = list(filter(lambda i: split <= int(i.split('_')[4]) < split2,files_all))
            files_test = list(filter(lambda i: int(i.split('_')[4]) >= split2,files_all))
        #files_test = list(filter(lambda i: split <= int(i.split('_')[4]),files_all))
        #files_val = files_test[:]
        #files_val = files_test_all[:int(len(files_test_all)/2)]
        #files_test = files_test_all[int(len(files_test_all)/2):]
        random.shuffle(files_train)
        random.shuffle(files_val)
        random.shuffle(files_test)
        templ = list(set([int(i.split('_')[0]) for i in files_val]))
        templ.sort()

        opts.feature_name = 'chirp_new_SpF'
    elif opts.data_format == 0:
        with open(os.path.join(opts.data_dir, 'cache','train_test_split.pkl'),'rb') as g:
            files_train,files_test_all = pickle.load(g)
        random.shuffle(files_train)
        files_val = files_test_all[:int(len(files_test_all)/2)]
        files_test = files_test_all[int(len(files_test_all)/2):]
        random.shuffle(files_val)
        random.shuffle(files_test)
    else:
        print('DATA SPLIT FOR SF8_UPLOAD NOT PREPARED')
        raise NotImplementedError
    print('TRAINING   DATASET SAMPLES CNT:',len(files_train),'\n',
          'VALIDATION DATASET SAMPLES CNT:',len(files_val),'\n',
          'TESTING    DATASET SAMPLES CNT:',len(files_test))

    training_dataset = lora_dataset(opts, files_train)
    val_dataset = lora_dataset(opts, files_val)
    testing_dataset = lora_dataset(opts, files_test)

    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,  drop_last=True)
    val_dloader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,  drop_last=True)
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,  drop_last=True)
    return training_dloader,val_dloader,testing_dloader



