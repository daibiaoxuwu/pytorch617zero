# data_loader.py

import os
import random
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import pickle

class lora_dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, opts, files_list, folders_list):
        self.opts = opts
        self.data_lists = files_list
        self.folders_list = folders_list

    def __len__(self):
        return len(self.data_lists) 

    def load_img(self, path):
        lora_img = np.array(scio.loadmat(path)[self.opts.feature_name].tolist())
        lora_img = np.squeeze(lora_img)
        return torch.tensor(lora_img, dtype=torch.cfloat)


    def __getitem__(self, index0):
        for index in range(index0, index0 + len(self.data_lists)):
            data_file_name = self.data_lists[index0 % len(self.data_lists)]

            path = os.path.join(self.opts.data_dir, random.choice(self.folders_list), data_file_name)
            data_perY = self.load_img(path).cuda()

            data_file_parts = data_file_name.split('_')

            data_file_parts[1] = str(random.choice(self.opts.snr_list))

            data_pers = []
            for folder in self.folders_list:
                path = os.path.join(self.opts.data_dir, folder, '_'.join(data_file_parts))
                data_pers.append(self.load_img(path))
            data_pers = torch.stack(data_pers).cuda()

            label_per = (128-int(data_file_parts[0]))%128
            label_per = torch.tensor(label_per, dtype=int).cuda()

            return data_pers, label_per, data_perY
        raise StopIteration 


# receive the csi feature map derived by the ray model as the input
def lora_loader(opts):
    """Creates training and test data loaders.
    """
    print('creating dataloader',opts.snr_list)
    folders = os.listdir(opts.data_dir)[:3]
    files = os.listdir(os.path.join(opts.data_dir, folders[0]))
    files_filtered = list(filter(lambda i: i.split('_')[0] == i.split('_')[5] and i.split('_')[1] == '35', files))
    for folder in folders[1:]:
        files_filtered = list(filter(lambda i: os.path.exists(os.path.join(opts.data_dir, folder, i)), files_filtered))

    dpath='/data/djl/sf7-1b-split.pkl'
    if os.path.exists(dpath):
        print('reusing split from', dpath)
        with open(dpath,'rb') as g: 
            files_train,files_test = pickle.load(g)
    else:
        num_files = len(files_filtered)
        num_train = int(num_files * opts.ratio_bt_train_and_test)
        files_train = files_filtered[0:num_train]
        files_test = files_filtered[num_train:num_files]
        print("length of training and testing data is {},{}".format(len(files_train), len(files_test)))
        with open(dpath,'wb') as g: 
            pickle.dump([files_train,files_test], g)


    #random.shuffle(files_filtered)

    training_dataset = lora_dataset(opts, files_train, folders)
    testing_dataset = lora_dataset(opts, files_train, folders)

    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    print('finish')
    return training_dloader,testing_dloader


