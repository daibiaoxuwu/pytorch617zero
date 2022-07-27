import os 
import random 
import scipy.io as scio 
import numpy as np 
import torch 
from torch.utils.data import DataLoader 
from torch.utils import data 
import pickle 
import sys 
from utils import *
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
                if(re.search(r"SF\d+_125K", self.opts.data_dir)):
                    while(len(self.files[symbol_index]) < self.opts.stack_imgs):  
                        symbol_index = random.randint(0,self.opts.n_classes-1) 
                else:
                    while(len(self.files[symbol_index]) < 1):  
                        symbol_index = random.randint(0,self.opts.n_classes-1) 

                if(re.search(r"SF\d+_125K", self.opts.data_dir)):
                    fs = random.sample(self.files[symbol_index], self.opts.stack_imgs) 
                else:
                    fs = [random.choice(self.files[symbol_index]),]
                    for i in range(2,self.opts.stack_imgs+1):
                        fs.append( fs[0].replace('Gateway1', 'Gateway'+str(i))[:-1]+str(i) )
                        if not os.path.exists(fs[-1]): 
                            print(fs[-1])
                            return self.__getitem__(index0) 

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

                        #gen idea symbol
                        nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw) 
                        t = np.linspace(0, nsamp / self.opts.fs, nsamp) 
                        chirpI1 = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=90) 
                        chirpQ1 = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=0) 
                        mchirp = chirpI1+1j*chirpQ1 
                        mchirp = np.tile(mchirp, 2)
                        time_shift = round(symbol_index / self.opts.n_classes * nsamp)
                        time_shift = round((self.opts.n_classes - symbol_index) / self.opts.n_classes * nsamp)
                        chirp_ideal = torch.tensor(mchirp[time_shift:time_shift+nsamp],dtype=torch.cfloat)

                        images_X_spectrum_ideal = torch.stft(input=chirp_ideal,n_fft=self.opts.stft_nfft,win_length =self.opts.stft_nfft//self.opts.stft_mod, hop_length =int(self.opts.stft_nfft/32), return_complex=True).cuda()
                        ideal_symbol = torch.squeeze(spec_to_network_input2( spec_to_network_input(images_X_spectrum_ideal.unsqueeze(0), self.opts), self.opts )).cpu()

                        images_X_spectrum_raw = torch.stft(input=chirp_raw,n_fft=self.opts.stft_nfft,win_length =self.opts.stft_nfft//self.opts.stft_mod, hop_length =int(self.opts.stft_nfft/32), return_complex=True).cuda()
                        fake_symbol = torch.squeeze(spec_to_network_input2( spec_to_network_input(images_X_spectrum_raw.unsqueeze(0), self.opts), self.opts )).cpu()
                        
                        #print(fake_symbol[0].shape)
                        loss = torch.nn.MSELoss(reduction='mean')(torch.abs(ideal_symbol[0]+1j*ideal_symbol[1]), torch.abs(fake_symbol[0]+1j*fake_symbol[1]))
                        if loss>0.00025:
                            return self.__getitem__(index0) 

 
                        mwin = nsamp//2; 
                        datain = to_data(chirp_raw) 
                        A = uniform_filter1d(abs(datain),size=mwin) 
                        datain = datain[A >= max(A)/2] 
                        amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
                        chirp_raw /= amp_sig
                        #print(amp_sig) 
                        #assert(abs(amp_sig-0.04)<0.03) 
                         
                        amp = math.pow(0.1, snr/20) 
                        nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
                        noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat).cuda() 
                        data = (chirp_raw + noise)
 
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
 
    if(re.search(r"SF\d+_125K", opts.data_dir)):
        for ff in os.listdir(opts.data_dir): 
            for f in os.listdir(os.path.join(opts.data_dir, ff, 'woCFO')): 
                symbol_idx = (opts.n_classes - round(float(f.split('_')[1])))%opts.n_classes 
                files[symbol_idx].append(os.path.join(opts.data_dir, ff, 'woCFO', f)) 
    else:
        if max(opts.snr_list)!=min(opts.snr_list): raise  NotImplementedError('only single snr now')
        for i in range(1,5): assert 'Gateway'+str(i) in os.listdir(opts.data_dir), 'cannot find folder Gateway'+str(i)

        filelist = [ set(os.listdir(os.path.join(opts.data_dir, 'Gateway'+str(i)))) for i in range(2,5)]

        '''
        a = [0]*50
        for f in os.listdir(os.path.join(opts.data_dir, 'Gateway1')): 
            snr = round(float(f.split('_')[1]))
            a[-snr]+=1
        print(a)'''
        for ff in os.listdir(os.path.join(opts.data_dir, 'Gateway1')): 
            for f in os.listdir(os.path.join(opts.data_dir, 'Gateway1',ff, 'woCFO')): 
                symbol_idx = (opts.n_classes - round(float(f.split('_')[2])))%opts.n_classes 
                #snr = round(float(f.split('_')[1]))
                #if snr not in opts.snr_list: continue
                flag = 0
                pathf = os.path.join(opts.data_dir, 'Gateway1',ff, 'woCFO', f)
                for i in range(3): 
                    if not os.path.exists(pathf.replace('Gateway1', 'Gateway'+str(i+2))[:-1]+str(i+2)):
                        flag+=1
                        break
                if flag==0: files[symbol_idx].append(pathf)


    for i in files.keys(): 
        files[i].sort(key = lambda x: int(os.path.basename(x).split('_')[0])) 
    splitpos = [ opts.stack_imgs if len(files[i]) >= 2*opts.stack_imgs else 0 for i in range(opts.n_classes)] 

    a = [len(files[i]) for i in range(opts.n_classes)] 
    print('read data: max cnt', max(a), a.index(max(a)), 'min cnt', min(a), a.index(min(a)) )
 
    training_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][splitpos[i]:] for i in range(opts.n_classes)])) ) 
    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    testing_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][:splitpos[i]] for i in range(opts.n_classes)])) ) 
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    return training_dloader, testing_dloader 
 
 
 
 

