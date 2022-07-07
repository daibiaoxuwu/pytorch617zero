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

from torch.nn.functional import relu, max_pool2d
def norm(y):
    for i in range(y.shape[1]):
        y_abs = torch.abs(y[:,i])
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y[:,i] = torch.div(y[:,i], y_abs_max)
    return y
from torch.nn.functional import dropout2d
def complex_dropout2d(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    # this not a clean solution!
    mask = torch.ones(*input.shape, dtype = torch.float32)
    mask = dropout2d(mask, p, training)*1/(1-p)
    mask=mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    mask.type(input.dtype)
    return mask*input
def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output
def complex_max_pool2d(input,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    absolute_value, indices =  max_pool2d(
                               input.abs(), 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding, 
                               dilation = dilation,
                               ceil_mode = ceil_mode, 
                               return_indices = True
                            )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    abss = torch.abs(input)+1e-6
    angle = input / abss
    
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * angle
class classificationHybridModel3(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

     
    
    def __init__(self, conv_dim_in=2, conv_dim_out=128, conv_dim_lstm=1024): 
        super(classificationHybridModel3, self).__init__() 
 
        self.out_size = 1 
        self.conv1 = ComplexConv2d(1, 16, 3, 2, 1) 
        self.dense = ComplexLinear(conv_dim_lstm * 4, conv_dim_out * 2) 
        self.fcn1 = ComplexLinear(conv_dim_out * 2, conv_dim_out) 
        self.fcn2 = ComplexLinear(conv_dim_out, conv_dim_out) 
 
        self.act = complex_relu
 
    def forward(self, x): 
        out = self.act(self.conv1(x)) 
        out = complex_max_pool2d(out, 2, 2) 
        out = out.view(out.size(0), -1) 
 
        out = self.act(self.dense(out)) 
        out = complex_dropout2d(out, 0.1, self.training) 
 
        out = self.act(self.fcn1(out)) 
        out = complex_dropout2d(out, 0.1, self.training) 
        out = self.fcn2(out) 
        out = torch.softmax(abs(out),dim=1)
        return out 
 

class maskCNNModel3(nn.Module):
    def __init__(self, opts):
        super(maskCNNModel3, self).__init__()
        self.opts = opts

        self.conv1 = ComplexConv2d(1, 256, 5, padding=2)
        self.bn2d1 = ComplexBatchNorm2d(256)

        self.conv21a = ComplexConv2d(256*2+1, 64, 1)
        self.bn2d21a = ComplexBatchNorm2d(64)
        self.conv21b = ComplexConv2d(64, 64, 3, padding=1)
        self.bn2d21b = ComplexBatchNorm2d(64)
        self.conv21c = ComplexConv2d(64, 256, 1)
        self.bn2d21c = ComplexBatchNorm2d(256)
        self.conv22a = ComplexConv2d(256*2+1, 64, 1)
        self.bn2d22a = ComplexBatchNorm2d(64)
        self.conv22b = ComplexConv2d(64, 64, 3, padding=1)
        self.bn2d22b = ComplexBatchNorm2d(64)
        self.conv22c = ComplexConv2d(64, 256, 1)
        self.bn2d22c = ComplexBatchNorm2d(256)

        self.conv3a = ComplexConv2d(256, 64, 5, padding=2)
        self.bn2d3a = ComplexBatchNorm2d(64)
        self.conv3b = ComplexConv2d(64, 64, 5, padding=2)
        self.bn2d3b = ComplexBatchNorm2d(64)
        self.conv3c = ComplexConv2d(64, 8, 1)
        self.bn2d3c = ComplexBatchNorm2d(8)

        self.fc1 = ComplexLinear(opts.conv_dim_lstm , opts.fc1_dim)
        self.fc2 = ComplexLinear(opts.fc1_dim, opts.freq_size)

    def forward(self, xs):

        xs = [x.transpose(1,2).unsqueeze(1) for x in xs]
                
        # CNN_1
        outs = [self.conv1(x) for x in xs]
        outs = [self.bn2d1(x) for x in outs]
        outs = [complex_relu(x) for x in outs]

        # CNN_2
        '''
        out_abs = torch.mean(torch.stack([torch.abs(out) for out in outs],0),0)
        outavg = [out * (out_abs / torch.abs(out)) for out in outs]
        out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0]
        outmax = [out * (out_max / torch.abs(out)) for out in outs]
        out_min = torch.min(torch.stack([torch.abs(out) for out in outs],0),0)[0]
        outmin = [out * (out_min / torch.abs(out)) for out in outs]
        '''
        out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0]
        outmax = [out * (out_max / (torch.abs(out)+1e-6)) for out in outs]

        outs = [torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)]

        outs = [self.conv21a(x) for x in outs]
        outs = [self.bn2d21a(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv21b(x) for x in outs]
        outs = [self.bn2d21b(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv21c(x) for x in outs]
        outs = [self.bn2d21c(x) for x in outs]
        outs = [complex_relu(x) for x in outs]

        out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0]
        outmax = [out * (out_max /  (torch.abs(out)+1e-6)) for out in outs]

        outs = [torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)]

        outs = [self.conv22a(x) for x in outs]
        outs = [self.bn2d22a(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv22b(x) for x in outs]
        outs = [self.bn2d22b(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv22c(x) for x in outs]
        outs = [self.bn2d22c(x) for x in outs]
        outs = [complex_relu(x) for x in outs]

        # CNN_3
        outs = [self.conv3a(x) for x in outs]
        outs = [self.bn2d3a(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv3b(x) for x in outs]
        outs = [self.bn2d3b(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.conv3c(x) for x in outs]
        outs = [self.bn2d3c(x) for x in outs]
        outs = [complex_relu(x) for x in outs]

        outs = [x.transpose(1, 2) for x in outs]
        outs = [x.reshape(x.size(0), x.size(1), -1) for x in outs]
        outs = [self.fc1(x) for x in outs]
        outs = [complex_relu(x) for x in outs]
        outs = [self.fc2(x) for x in outs]
        outs = [complex_relu(x) for x in outs]

        #Final
        outs = [x.view(x.size(0), x.size(1), 1, -1) for x in outs]
        outs = [x.transpose(1, 2).transpose(2, 3).contiguous() for x in outs]

        return outs
