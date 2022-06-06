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
                ####DEBUG
                if self.opts.cut_data_by >= 2:
                    if self.initFlag < 3:
                        self.initFlag+=1
                        print('==========WARNING: USING PARTIAL DATA SYMBOL%',self.opts.cut_data_by,'==1 ========')
                        if self.opts.SpFD == 'True':
                            print('==========WARNING: USING ONLY SNR_LIST[0] ========')
                    ### CUT DATA
                    if not label_per % self.opts.cut_data_by==1: continue

                if self.opts.data_format == 3:
                    paths = [os.path.join(self.opts.data_dir, folder, data_file_name) for folder in self.folders_list]
                    data_perY = [self.load_img(path).cuda() for path in paths]
                elif self.opts.data_format == 2: #DEBUG!!! LOAD -15 INSTEAD OF 35 FOR FORMAT==2
                    data_file_parts[1] = '35'
                    data_file_parts[4] = '0'
                    data_file_parts[6] = '1'
                    data_file_parts[7] = '1.mat'
                    data_file_name_new = '_'.join(data_file_parts)
                    path = os.path.join(self.opts.data_dir, data_file_name_new)
                    data_perY = [self.load_img(path).cuda() for i in range(self.opts.stack_imgs)]

                elif self.opts.data_format < 2:
                    path = os.path.join(self.opts.data_dir, data_file_name)
                    if self.opts.SpFD == 'False':
                        data_perY = [self.load_img(path).cuda() for i in range(self.opts.stack_imgs)]
                    elif self.opts.SpFD == 'True': # a single 1MHz sampling rate, split to 4 * 250KHz
                        assert self.opts.data_dir == 0
                        data_perY_orig = self.load_img(path).cuda()
                        data_perY = [torch.zeros(data_perY_orig.shape[0]//self.opts.stack_imgs, dtype=torch.cfloat) for i in range(self.opts.stack_imgs)]
                        for i in range(data_perY[0].shape[0]):
                            for idxi, data in enumerate(data_perY): data[i] = data_perY_orig[i * self.opts.stack_imgs + idxi]
                    else: raise NotImplementedError
                else: raise NotImplementedError

                
                if self.opts.SpFD == 'False':
                    data_pers = []
                    for k in range(self.opts.stack_imgs):
                        if self.opts.same_img == 'False': # do not use the SNR-15 generated from the same SNR35 image as dataY
                            index_input = index + 1
                            while index_input < index + len(self.data_lists):
                                data_file_name = self.data_lists[index_input % len(self.data_lists)]
                                data_file_parts = data_file_name.split('_')
                                label_input = int(data_file_parts[0].split('/')[-1])
                                if(label_input == label_per.item()  ):
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
                                    #print(index_input, data_file_name, label_per.item())
                                    index_input += 1
                            if index_input == index0 + len(self.data_lists): raise StopIteration
                        else: raise NotImplementedError
                elif self.opts.SpFD == 'True': # a single 1MHz sampling rate, split to 4 * 250KHz
                            index_input = index + 1
                            while index_input < index + len(self.data_lists):
                                data_file_name = self.data_lists[index_input % len(self.data_lists)]
                                data_file_parts = data_file_name.split('_')
                                label_input = int(data_file_parts[0].split('/')[-1])
                                if(label_input == label_per):
                                    if self.opts.data_format == 0:
                                        snr = str(self.opts.snr_list[0]) ## USE A SINGLE SNR
                                        data_part0 = data_file_parts[0].split('/')
                                        data_part0[-2] = snr
                                        data_file_parts[0] = '/'.join(data_part0)
                                        data_file_parts[1] = snr 
                                        if self.opts.random_idx == 'False':
                                            data_file_parts[-1] = str((index_input) % 100 + 1) + '.mat'
                                        else:
                                            data_file_parts[-1] = str(random.randint(1,100)) + '.mat'
                                        data_file_name_new = '_'.join(data_file_parts)
                                        path = os.path.join(self.opts.data_dir, data_file_name_new)
                                        data_per_orig = self.load_img(path)


                                        data_pers = [torch.zeros(data_per_orig.shape[0]//self.opts.stack_imgs, dtype=torch.cfloat) for i in range(self.opts.stack_imgs)]
                                        for i in range(data_pers[0].shape[0]):
                                            for idxi in range(self.opts.stack_imgs):
                                                data_pers[idxi][i] = data_per_orig[i * self.opts.stack_imgs + idxi]

                                        '''
                                        images_X_SpF_spectrum_raw = torch.stft(input= data_pers[0], n_fft=self.opts.stft_nfft,
                                                        hop_length=self.opts.stft_overlap // self.opts.stack_imgs, win_length=self.opts.stft_window // self.opts.stack_imgs,
                                                        pad_mode='constant')
                                        freq_size = self.opts.freq_size
                                        # trim
                                        trim_size = freq_size // 2
                                        # up down 拼接
                                        images_X_SpF_spectrum_raw = torch.cat((images_X_SpF_spectrum_raw[-trim_size:, :], images_X_SpF_spectrum_raw[0:trim_size, :]), 0)
                                        print(images_X_SpF_spectrum_raw.shape,'images_X_SpF_spectrum_raw')

                                        merged = to_data(np.abs(images_X_SpF_spectrum_raw))
                                        merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged)) * 255
                                        merged = np.squeeze(merged)
                                        merged = cv2.flip(merged, 0)
                                        cv2.imwrite('SpFData'+str(self.opts.stack_imgs)+'.png', merged)
                                        print('SpFData')
                                        sys.exit(1)'''

                                    else: raise NotImplementedError
                                    data_file_names.append(data_file_name_new)
                                    break
                                index_input += 1
                                if index_input == index0 + len(self.data_lists): raise StopIteration

                else: raise NotImplementedError
                    

                ### ABOUT SPF
                '''
                if self.opts.data_dir == '/data/djl/SpF102' and index0<5:
                    data_SpF_orig = torch.zeros(data_pers[0].shape[0]*self.opts.stack_imgs, dtype=torch.cfloat)
                    print('data_pers',data_pers[0].shape)
                    for i in range(data_pers[0].shape[0]):
                        for j in range(self.opts.stack_imgs):
                            data_SpF_orig[i*self.opts.stack_imgs+j] = data_pers[j][i]
                    images_X_SpF_spectrum_raw = torch.stft(input=data_SpF_orig, n_fft=self.opts.stft_nfft,
                                                                hop_length=self.opts.stft_overlap, win_length=self.opts.stft_window,
                                                                pad_mode='constant').cuda()
                    print(images_X_SpF_spectrum_raw.dtype)
                    images_X_SpF_spectrum_raw = torch.unsqueeze(images_X_SpF_spectrum_raw,0)
                    print(images_X_SpF_spectrum_raw.shape)
                    images_X_SpF_spectrum = to_data(spec_to_network_input( images_X_SpF_spectrum_raw, self.opts))
                    print('111')
                    merged = np.abs(images_X_SpF_spectrum[:, 0, :, :]+1j*images_X_SpF_spectrum[:, 1, :, :])
                    print(merged.shape)
                    merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged)) * 255
                    merged = np.squeeze(merged)
                    merged = cv2.flip(merged, 0)
                    print('222')
                    cv2.imwrite('SpFOrig'+str(index0)+'.png', merged)
                    sys.exit(1)'''

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
    """Creates training and test data loaders.
    code for loading sf7-1b-out-upload:
        folders = os.listdir(opts.data_dir)[:opts.stack_imgs]
        files = os.listdir(os.path.join(opts.data_dir, folders[0]))
        files_filtered = list(filter(lambda i: i.split('_')[0] == i.split('_')[5] and i.split('_')[1] == '35', files))
        for folder in folders[1:]:
            files_filtered = list(filter(lambda i: os.path.exists(os.path.join(opts.data_dir, folder, i)), files_filtered))
        random.shuffle(files_filtered)
    code for reading /data/djl/data0306/data:
        files_filtered = []
        for i in range(128):
            filelist = os.listdir(os.path.join(opts.data_dir,str(i),str(opts.groundtruth_code))) #use SNR<-15 for training data(X); SNR=+35 for groundtruth(Y). we first find the SNR=+35 data (Y) and then read the corresponding SNR<-15 data for training(X).
            files_filtered.extend([ os.path.join(opts.data_dir,str(i),str(opts.groundtruth_code),j) for j in filelist])
        random.shuffle(files_filtered)
    code for split and dump:
        num_files = len(files_filtered)
        num_train = int(num_files * opts.ratio_bt_train_and_test)
        files_train = files_filtered[0:num_train]
        files_test = files_filtered[num_train:num_files]
        print("length of training and testing data is {},{}".format(len(files_train), len(files_test)))
        with open(dpath,'wb') as g: 
            pickle.dump([files_train,files_test], g)
    """
    if opts.data_format == 1:
        print('WARNING, USING ARBITARY PARTITION FOR', opts.data_dir, 'AND CHANGING opts.feature_name = chirp_new_SpF')

        files = os.listdir(os.path.join(opts.data_dir))
        files_train = list(filter(lambda i: i.split('_')[1] == '35' and int(i.split('_')[4]) < 90 and i.split('_')[7].split('.')[0] == '1', files))
        files_test = list(filter(lambda i: i.split('_')[1] == '35' and int(i.split('_')[4]) >= 90 and i.split('_')[7].split('.')[0] == '1', files))
        random.shuffle(files_train)
        random.shuffle(files_test)
        opts.feature_name = 'chirp_new_SpF'
    elif opts.data_format == 2:
        print('WARNING, USING ARBITARY PARTITION FOR', opts.data_dir, 'AND CHANGING opts.feature_name = chirp_new_SpF')
        files = os.listdir(os.path.join(opts.data_dir))

        snr_all = set( [int(i.split('_')[1]) for i in files])
        print('ALL SNR IN DATASET:',snr_all)
        main_snr = opts.snr_list[0]
        files_35 = list(filter(lambda i: i.split('_')[1] == str(35) ,files))
        vals_35 = set([i.split('_')[0] for i in files_35])
        print('VAL COUNT', len(vals_35))
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
                
        templ = list(set([int(i.split('_')[4]) for i in files]))
        #templ.sort()
        #for k in templ: print(k,len(set(filter(lambda i: int(i.split('_')[4]) == k,files))))
        for i in opts.snr_list: 
            if i not in snr_all: 
                print('SNR LIST FAILURE',opts.snr_list, i, snr_all)
                sys.exit(1)
        print(templ)
        split = int(max(templ))
        files_train = list(files_all)
        files_test = list(files_all)
        files_val = list(files_all)

        #split2 = int(max(templ)*0.96)
        #files_train = list(filter(lambda i: int(i.split('_')[4]) < split,files_all))
        #files_val = list(filter(lambda i: split <= int(i.split('_')[4]) < split2,files_all))
        #files_test = list(filter(lambda i: int(i.split('_')[4]) >= split2,files_all))
        #files_test = list(filter(lambda i: split <= int(i.split('_')[4]),files_all))
        #files_val = files_test[:]
        #files_val = files_test_all[:int(len(files_test_all)/2)]
        #files_test = files_test_all[int(len(files_test_all)/2):]
        random.shuffle(files_train)
        random.shuffle(files_val)
        random.shuffle(files_test)
        templ = list(set([int(i.split('_')[0]) for i in files_val]))
        templ.sort()
        print(len(templ),'1111')


        print(files_val[0])
        opts.feature_name = 'chirp_new_SpF'
    elif opts.data_format == 0:
        with open(os.path.join(opts.data_dir, 'cache','train_test_split.pkl'),'rb') as g:
            files_train,files_test_all = pickle.load(g)
        random.shuffle(files_train)
        files_val = files_test_all[:int(len(files_test_all)/2)]
        files_test = files_test_all[int(len(files_test_all)/2):]
        random.shuffle(files_val)
        random.shuffle(files_test)
        files_val = files_val[:1600]
        files_test = files_test[:1600]
    else:
        print('DATA SPLIT FOR SF8_UPLOAD NOT PREPARED')
        raise NotImplementedError
    print('TRAINING   DATASET S35 SAMPLES CNT:',len(files_train),'\n',
          'VALIDATION DATASET S35 SAMPLES CNT:',len(files_val),'\n',
          'TESTING    DATASET S35 SAMPLES CNT:',len(files_test))

    training_dataset = lora_dataset(opts, files_train)
    val_dataset = lora_dataset(opts, files_val)
    testing_dataset = lora_dataset(opts, files_test)

    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_dloader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    return training_dloader,val_dloader,testing_dloader



