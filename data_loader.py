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
import re 
from scipy.ndimage.filters import uniform_filter1d 
 
class lora_dataset(data.Dataset): 
    'Characterizes a dataset for PyTorch' 
 
    def __init__(self, opts, files): 
        self.opts = opts 
        self.files = files 
 
    def __len__(self): 
        return np.iinfo(np.int64).max 
 
    def load_img(self, path): 
        fid = open(path, 'rb') 
        nelements = self.opts.n_classes * self.opts.fs // self.opts.bw 
        lora_img = np.fromfile(fid, np.float32, nelements * 2) 
        lora_img = lora_img[::2] + lora_img[1::2]*1j 
        return torch.tensor(lora_img) 
 
    def __getitem__(self, index0): 
            #if index0 == 0: print('=========using real data and adding noise according to signal amplitude') 
            try: 
                data_perY = [] 
                symbol_index = random.randint(0,self.opts.n_classes-1) 
                while(len(self.files[symbol_index]) < self.opts.stack_imgs):  
                    symbol_index = random.randint(0,self.opts.n_classes-1) 
                fs = random.sample(self.files[symbol_index], self.opts.stack_imgs) 
                for k in range(self.opts.stack_imgs): 
                    chirp_raw = self.load_img(fs[k]).cuda() 
                    data_perY.append(chirp_raw) 
 
                data_pers = [] 
                for k in range(self.opts.stack_imgs): 
                        snr = self.opts.snr_list[k] 
 
                        nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw) 
                        t = np.linspace(0, nsamp / self.opts.fs, nsamp) 
                        chirpI1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=90) 
                        chirpQ1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=0) 
                        chirp_down = chirpI1+1j*chirpQ1 
                         
                        chirp_raw = data_perY[k] 
                        phase = random.uniform(-np.pi, np.pi) 
                        chirp_raw *= (np.cos(phase)+1j*np.sin(phase)) 
                        ''' 
                        chirp_dechirp = chirp_raw .* chirp_down 
                        chirp_fft_raw =abs(np.fft.fft(chirp_dechirp, nsamp*10)) 
                        align_win_len = len(chirp_fft_raw) / (self.opts.fs/self.opts.bw); 
 
                        chirp_fft_overlap=chirp_fft_raw[:align_win_len]+chirp_fft_raw[-align_win_len:] 
                        chirp_peak_overlap=abs(chirp_fft_overlap) 
                        [pk_height_overlap,pk_index_overlap]=max(chirp_peak_overlap)''' 
 
                        mwin = nsamp//2; 
                        datain = to_data(chirp_raw) 
                        A = uniform_filter1d(abs(datain),size=mwin) 
                        datain = datain[A >= max(A)/2] 
                        amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
                        #print(amp_sig) 
                        #assert(abs(amp_sig-0.04)<0.03) 
                         
                        amp = amp_sig*math.pow(0.1, snr/20) 
                        nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
                        noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat).cuda() 
                        data = chirp_raw + noise 
 
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
                print(e) 
            except OSError as e: 
                print(e) 
 
 
 
# receive the csi feature map derived by the ray model as the input 
def lora_loader(opts): 
    files = dict(zip(list(range(opts.n_classes)), [[] for i in range(opts.n_classes)])) 
 
    assert(re.search(r"SF\d+_125K", opts.data_dir)) 
    for ff in os.listdir(opts.data_dir): 
        for f in os.listdir(os.path.join(opts.data_dir, ff, 'woCFO')): 
            symbol_idx = (opts.n_classes - round(float(f.split('_')[1])))%opts.n_classes 
            files[symbol_idx].append(os.path.join(opts.data_dir, ff, 'woCFO', f)) 
    for i in files.keys(): 
        files[i].sort(key = lambda x: int(os.path.basename(x).split('_')[0])) 
    splitpos = [ opts.stack_imgs if len(files[i]) >= 2*opts.stack_imgs else 0 for i in range(opts.n_classes)] 
     
 
    training_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][splitpos[i]:] for i in range(opts.n_classes)])) ) 
    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    testing_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][:splitpos[i]] for i in range(opts.n_classes)])) ) 
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    return training_dloader, testing_dloader 
 
 
 
 

