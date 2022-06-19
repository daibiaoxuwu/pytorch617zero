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
                    data_file_parts[1] = '35'
                    data_file_parts[4] = '0'
                    data_file_parts[6] = '1'
                    data_file_parts[7] = '1.mat'
                    data_file_name_new = '_'.join(data_file_parts)
                    path = os.path.join('/data/djl/sf'+str(self.opts.sf)+'_125k_new', data_file_name_new)
                    data_perY = [self.load_img(path).cuda() for i in range(self.opts.stack_imgs)]

                    nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw)
                    t = np.linspace(0, nsamp / self.opts.fs, nsamp)
                    phi = random.randint(-90, 90)
                    chirpI = chirp(t, f0=-self.opts.bw, f1=self.opts.bw, t1=nsamp / self.opts.fs, method='linear', phi=phi+90)
                    chirpQ = chirp(t, f0=-self.opts.bw, f1=self.opts.bw, t1=nsamp / self.opts.fs, method='linear', phi=phi)
                    plt.plot(t, chirpI)
                    plt.plot(t, chirpQ)
                    plt.savefig('1.png')
                    plt.clf()
                    plt.plot(t, to_data(data_perY[1]).real)
                    plt.savefig('2.png')
                    mchirp = chirpI+1j*chirpQ
                    mchirp = np.repeat(mchirp, 2, axis=0)
                    symbol_index = ( int(data_file_parts[5])+self.opts.n_classes //2)%self.opts.n_classes
                    time_shift = round((self.opts.n_classes - symbol_index) / self.opts.n_classes * nsamp)
                    print(time_shift)
                    chirp_raw = mchirp[time_shift:time_shift+nsamp]
                    plt.plot(t, chirp_raw.real)
                    plt.savefig('3.png')
                    sys.exit(1)
                    data_perY[0] = torch.tensor(chirp_raw).cuda()



                elif self.opts.data_format < 2:
                    path = os.path.join(self.opts.data_dir, data_file_name)
                    data_perY = [self.load_img(path).cuda() for i in range(self.opts.stack_imgs)]

                data_pers = []
                for k in range(self.opts.stack_imgs):
                        index_input = index + 1
                        while index_input < index + len(self.data_lists):
                            data_file_name = self.data_lists[index_input % len(self.data_lists)]
                            data_file_parts = data_file_name.split('_')
                            label_input = int(data_file_parts[0].split('/')[-1])
                            if(label_input == label_per.item()  ):
                                #print('ImgA',k, data_file_name)
                                if self.opts.data_format == 3:
                                    data_file_parts[1] = str(random.choice(self.opts.snr_list))
                                    data_file_name_new = '_'.join(data_file_parts)
                                    path = os.path.join(self.opts.data_dir, self.folders_list[k], data_file_name_new)
                                    data_pers.append(self.load_img(path))
                                elif self.opts.data_format == 1 or self.opts.data_format == 2:
                                    data_file_parts[1] = str(random.choice(self.opts.snr_list))
                                    if self.opts.data_format == 1: data_file_parts[7] = str(k + 1) + '.mat'
                                    data_file_name_new = '_'.join(data_file_parts)
                                    path = os.path.join(self.opts.data_dir, data_file_name_new)
                                    data_pers.append(self.load_img(path))
                                elif self.opts.data_format == 0:
                                    snr = str(self.opts.snr_list[k])
                                    data_part0 = data_file_parts[0].split('/')
                                    data_part0[-2] = snr
                                    data_file_parts[0] = '/'.join(data_part0)
                                    data_file_parts[1] = snr 
                                    if self.opts.random_idx == 'False':
                                        data_file_parts[-1] = str((index_input+k) % 100 + 1) + '.mat'
                                    else:
                                        data_file_parts[-1] = str(random.randint(1,100)) + '.mat'
                                    data_file_name_new = '_'.join(data_file_parts)
                                    path = os.path.join(self.opts.data_dir, data_file_name_new)
                                    data_pers.append(self.load_img(path))
                                else: raise NotImplementedError
                                data_file_names.append(data_file_name_new)
                                break
                            else:
                                index_input += 1

                        if index_input == index + len(self.data_lists): 
                            print('SEARCH FAILURE: INDEXINPUT', index_input, index + len(self.data_lists))
                            return
                

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



