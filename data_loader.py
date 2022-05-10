# data_loader.py

import os
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F
import random
import time


import scipy.io as scio
import numpy as np
from PIL import Image
import pickle

import math
import time
from utils import to_var

class lora_dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, opts, files_list, istrain,  filenames, answers):
        'Initialization'
        # self.scaling_for_intensity = opts.scaling_for_intensity
        self.featrue_name = opts.feature_name  # get from config.create_parser
        self.data_dir = opts.data_dir
        self.stack_imgs = opts.stack_imgs
        self.snr_list = opts.snr_list
        self.initcuda = False

        dpath = '/data/djl/test_dataset'+str(opts.snr_list)[1:-1]+str(opts.stack_imgs)+'.pkl'
        if istrain: dpath = '/data/djl/train_dataset'+str(opts.snr_list)[1:-1]+str(opts.stack_imgs)+'.pkl'
        if os.path.exists(dpath) and opts.use_old_data == 'True':
            with open(dpath,'rb') as f:
                [self.data2,self.label_pers, self.data_perYs] = pickle.load(f)
                self.data_perYs = self.data_perYs.cuda()
                self.data2 = self.data2.cuda()
                self.maxidx = self.data2.shape[0]
                print('load data max steps:', int(self.maxidx/opts.batch_size))
                stats = np.bincount(self.label_pers.cpu())
                print('dataset count of each of the', 2**opts.sf,' symbols: max', np.max(stats),'min', np.min(stats), 'avg',np.mean(stats))

        else:
            starts = [0]*128
            data2 = []
            label_pers = []
            data_perYs=[]
            label_perYs=[]
            for idx,i in enumerate(answers):
                d1 = []
                while True:
                    try:
                        starts[i]+=1
                        data_file_name0 = filenames[i][starts[i]]
                        lora_img = np.array(scio.loadmat(data_file_name0)[self.featrue_name].tolist())
                        break
                    except Exception as e:
                        print(e)
                        print(data_file_per)
                data_perY = np.squeeze(lora_img)
                label_perY = data_file_name0[:-4]
                data_per = []
                label_per = []
                for k in range(self.stack_imgs):
                    while True:
                        snr = random.choice(self.snr_list)
                        data_file_name0 = filenames[i][starts[i]]
                        starts[i]+=1
                        val = idx%100+1
                        data_file_path = data_file_name0.split("/")
                        assert(data_file_path[-2] == '35')
                        data_file_path[-2] = str(snr)
                        data_file_name = data_file_path[-1].split('_')
                        assert(data_file_name[1] == '35')
                        data_file_name[1] = str(snr)
                        data_file_name[-1] = str(val) +'.mat' 
                        data_file_name = ('_').join(data_file_name)
                        data_file_path[-1] = data_file_name
                        data_file_name = ('/').join(data_file_path)

                        data_file_per = os.path.join(self.data_dir, data_file_name)
                        try:
                            lora_img = np.array(scio.loadmat(data_file_per)[self.featrue_name].tolist())
                            lora_img = np.squeeze(lora_img)
                            data_per.append(lora_img)
                            label_per = int(data_file_name[:-4].split('_')[5])
                            break
                        except Exception as e:
                            print(e)
                            print(data_file_per)

                label_pers.append(label_per)
                data_perYs.append(data_perY)
                label_perYs.append(label_perY)
                data2.append(np.stack(data_per))
            self.data_perYs = torch.tensor(np.stack(data_perYs), dtype=torch.cfloat)
            self.data_perYs = self.data_perYs.cuda()
            self.data2 = torch.tensor(np.stack(data2), dtype=torch.cfloat)
            self.data2 = self.data2.cuda()
            self.label_pers =torch.tensor(label_pers,dtype=int).cuda()
            self.label_perYs = label_perYs
            self.maxidx = self.data2.shape[0]
            if opts.write_new_data == 'True':
                with open(dpath,'wb') as g: 
                    pickle.dump([self.data2,self.label_pers, self.data_perYs], g)

    def __len__(self):
        'Denotes the total number of samples'
        return self.maxidx

    def __getitem__(self, index):
        index = index % self.maxidx
        temp = [self.data2.select(0,index) , self.label_pers.select(0,index), self.data_perYs.select(0,index)]
        return temp

# receive the csi feature map derived by the ray model as the input
def lora_loader(opts):
    """Creates training and test data loaders.
    """
    dpath = '/data/djl/test_dataset'+str(opts.snr_list)[1:-1]+str(opts.stack_imgs)+'.pkl'
    dpath2 = '/data/djl/train_dataset'+str(opts.snr_list)[1:-1]+str(opts.stack_imgs)+'.pkl'
    if os.path.exists(dpath) and os.path.exists(dpath2) and opts.use_old_data == 'True':
        training_dataset =  lora_dataset(opts, [], True, [], [])
        testing_dataset =   lora_dataset(opts, [], False, [], [])
        training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        return training_dloader, testing_dloader
    else:
        print('processing training data and dumping to', dpath)
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

        random.shuffle(files_train)
        random.shuffle(files_test)

        filenames_train = [ [] for x in range(128)]
        for data_file_name0 in files_train:
            code_label  = int(data_file_name0.split("_")[5])
            filenames_train[code_label].append(data_file_name0)
        for i in range(128): random.shuffle(filenames_train[i])

        filenames_test = [ [] for x in range(128)]
        for data_file_name0 in files_test:
            code_label  = int(data_file_name0.split("_")[5])
            filenames_test[code_label].append(data_file_name0)
        for i in range(128): random.shuffle(filenames_test[i])

        answers = [i%128 for i in range(opts.train_datacnt * 128)]
        random.shuffle(answers)

        training_dataset =  lora_dataset(opts, files_train, True, filenames_train, answers)
        answers = [i%128 for i in range(opts.test_datacnt * 128)]
        random.shuffle(answers)
        testing_dataset =   lora_dataset(opts, files_test, False, filenames_test, answers)

        training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        return training_dloader, testing_dloader
