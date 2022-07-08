# models.py 
 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time 
import cv2 
import numpy as np 
import os 
from utils import * 
import math 
import sys   
 
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear 
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d 
 
from torch.nn.functional import relu, max_pool2d, tanh 
def complex_tanh(input): 
    return tanh(input.real).type(torch.complex64)+1j*tanh(input.imag).type(torch.complex64) 
 
class classificationHybridModel3(nn.Module): 
    """Defines the architecture of the discriminator network. 
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment. 
    """ 
 
    def __init__(self, conv_dim_in=2, conv_dim_out=128, conv_dim_lstm=1024):  
        super(classificationHybridModel3, self).__init__()  
  
        self.out_size = 1  
        self.conv1 = nn.Conv2d(2, 16, (3, 3), stride=(2, 2), padding=(1, 1))  
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))  
        self.dense = nn.Linear(conv_dim_lstm * 4, conv_dim_out * 2)  
        self.fcn1 = nn.Linear(conv_dim_out * 2, conv_dim_out)  
        self.fcn2 = nn.Linear(conv_dim_out, conv_dim_out)  
        self.softmax = nn.Softmax(dim=1)  
  
        self.drop1 = nn.Dropout(0.1)  
        self.drop2 = nn.Dropout(0.1)  
        self.act = nn.LeakyReLU()  
  
    def forward(self, x):  
        out = self.act(self.conv1(x))  
        out = self.pool1(out)  
        out = out.view(out.size(0), -1)  
  
        out = self.act(self.dense(out))  
        out = self.drop2(out)  
  
        out = self.act(self.fcn1(out))  
        out = self.drop1(out)  
        out = self.fcn2(out)  
        return out   
 
class maskCNNModel3(nn.Module): 
    def __init__(self, opts): 
        super(maskCNNModel3, self).__init__() 
        self.opts = opts 
 
        self.conv1 = ComplexConv2d(1, 64, 7, padding=3) 
        self.bn2d1 = ComplexBatchNorm2d(64) 
 
        self.conv21a = ComplexConv2d(64, 64, 5, padding=2) 
        self.bn2d21a = ComplexBatchNorm2d(64) 
        self.conv21b = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d21b = ComplexBatchNorm2d(64) 
        self.conv21c = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d21c = ComplexBatchNorm2d(64) 
        self.conv22b = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d22b = ComplexBatchNorm2d(64) 
        self.conv22c = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d22c = ComplexBatchNorm2d(64) 
 
        self.conv3a = ComplexConv2d(64, 64, 5, padding=2) 
        self.bn2d3a = ComplexBatchNorm2d(64) 
        self.conv3b = ComplexConv2d(64, 64, 7, padding=3) 
        self.bn2d3b = ComplexBatchNorm2d(64) 
        self.conv3c = ComplexConv2d(64, 1, 5, padding=2) 
        self.bn2d3c = ComplexBatchNorm2d(1) 
 
        #self.fc1 = ComplexLinear(opts.conv_dim_lstm , opts.fc1_dim) 
        #self.fc2 = ComplexLinear(opts.fc1_dim, opts.freq_size) 
 
    def forward(self, xslist): 
        outlist = []
        for xs in xslist:
            outs = (xs[:,0]+1j*xs[:,1]).unsqueeze(1) 
                     
            # CNN_1 
            outs = self.conv1(outs) 
            #outs = self.bn2d1(outs) 
            outs = complex_relu(outs) 
     
            # CNN_2 
            ''' 
            out_abs = torch.mean(torch.stack([torch.abs(out) for out in outs],0),0) 
            outavg = [out * (out_abs / torch.abs(out)) for out in outs] 
            out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
            outmax = [out * (out_max / torch.abs(out)) for out in outs] 
            out_min = torch.min(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
            outmin = [out * (out_min / torch.abs(out)) for out in outs] 
            ''' 
            ''' 
            out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
            outmax = [out * (out_max / (torch.abs(out)+1e-6)) for out in outs] 
     
            outs = torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)] 
            ''' 
     
            outs = self.conv21a(outs) 
            #outs = self.bn2d21a(outs) 
            outs = complex_relu(outs) 
            outs = self.conv21b(outs) 
            #outs = self.bn2d21b(outs) 
            outs = complex_relu(outs) 
            outs = self.conv21c(outs) 
            #outs = self.bn2d21c(outs) 
            outs = complex_relu(outs) 
     
            ''' 
            out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
            outmax = [out * (out_max /  (torch.abs(out)+1e-6)) for out in outs] 
     
            outs = torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)] 
            ''' 
     
            outs = self.conv3c(outs) 
            #outs = self.bn2d3c(outs) 
            #outs = complex_tanh(outs) 
     
            ''' 
            outs = x.transpose(1, 2) for x in outs] 
            outs = x.reshape(x.size(0), x.size(1), -1) for x in outs] 
            outs = self.fc1(outs) 
            outs = complex_relu(outs) 
            outs = self.fc2(outs) 
            outs = complex_relu(outs) 
     
            #Final 
            outs = x.view(x.size(0), x.size(1), 1, -1) for x in outs] 
            outs = x.transpose(1, 2).transpose(2, 3).contiguous() for x in outs]''' 
     
            outs = torch.stack(( outs.real, outs.imag), 1).squeeze()
            outlist.append(outs)
        return outlist

