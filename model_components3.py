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
from complexPyTorch.complexFunctions import complex_max_pool2d , complex_relu
import torchvision
from torch.nn.functional import relu, max_pool2d, tanh 
def complex_tanh(input): 
    return input / torch.abs(input) * tanh(torch.abs(input))

class classificationHybridModel3(nn.Module): 
    """Defines the architecture of the discriminator network. 
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment. 
    """ 
 
    def __init__(self, conv_dim_in=1, conv_dim_out=128, conv_dim_lstm=1024): 
        super(classificationHybridModel3, self).__init__()  
        self.resnet = torchvision.models.resnet18(num_classes = conv_dim_out)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, conv_dim_out)
        self.resnet.conv1 = nn.Conv2d(2,64,7,padding=3)

    def forward(self, x):  
        out = self.resnet(x)
        return out   
 
'''
class BasicBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv2d(129, 64, 3, padding=1) 
        self.bn1 = ComplexBatchNorm2d(64) 
        self.conv2 =  ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2 = ComplexBatchNorm2d(64) 

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = complex_tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = complex_tanh(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.


    def __init__(self):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = ComplexConv2d(129, 64, 1)
        self.bn1 = ComplexBatchNorm2d(64) 
        self.conv2 = ComplexConv2d(64, 64, 3)
        self.bn1 = ComplexBatchNorm2d(64) 
        self.conv3 = ComplexConv2d(64, 64, 1)
        self.bn3 = ComplexBatchNorm2d(64)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = complex_tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = complex_tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = complex_tanh(out)

        return out'''


class maskCNNModel3(nn.Module): 
    def __init__(self, opts): 
        super(maskCNNModel3, self).__init__() 
        self.opts = opts 
 
        self.conv1 = ComplexConv2d(1, 64, 7, padding=3) 
        self.bn2d1 = ComplexBatchNorm2d(64) 
 
        self.conv21a = ComplexConv2d(129, 64, 5, padding=2) 
        self.bn2d21a = ComplexBatchNorm2d(64) 
        self.conv21b = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d21b = ComplexBatchNorm2d(64) 
        self.conv21c = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d21c = ComplexBatchNorm2d(64) 
        self.conv22a = ComplexConv2d(129, 64, 5, padding=2) 
        self.bn2d22a = ComplexBatchNorm2d(64) 
        self.conv22b = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d22b = ComplexBatchNorm2d(64) 
        self.conv22c = ComplexConv2d(64, 64, 3, padding=1) 
        self.bn2d22c = ComplexBatchNorm2d(64) 
 
        self.conv3a = ComplexConv2d(129, 64, 5, padding=2) 
        self.bn2d3a = ComplexBatchNorm2d(64) 
        self.conv3b = ComplexConv2d(64, 64, 7, padding=3) 
        self.bn2d3b = ComplexBatchNorm2d(64) 
        self.conv3c = ComplexConv2d(64, 1, 5, padding=2) 
        self.bn2d3c = ComplexBatchNorm2d(1) 
 
        #self.fc1 = ComplexLinear(opts.conv_dim_lstm , opts.fc1_dim) 
        #self.fc2 = ComplexLinear(opts.fc1_dim, opts.freq_size) 
 
    def forward(self, xslist): 
        xs = [(xs[:,0]+1j*xs[:,1]).unsqueeze(1) for xs in xslist]
                 
        # CNN_1 
        outs = [self.conv1(x) for x in xs] 
        outs = [self.bn2d1(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs1 = outs
 
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
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.conv21b(x) for x in outs] 
        outs = [self.bn2d21b(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.conv21c(x) for x in outs] 
        outs = [self.bn2d21c(x) for x in outs] 
        outs = [complex_tanh(x+y) for x,y in zip(outs,outs1)] 
        outs2 = outs
 
        out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
        outmax = [out * (out_max / (torch.abs(out)+1e-6)) for out in outs] 
 
        outs = [torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)] 
 
        outs = [self.conv22a(x) for x in outs] 
        outs = [self.bn2d22a(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.conv22b(x) for x in outs] 
        outs = [self.bn2d22b(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.conv22c(x) for x in outs] 
        outs = [self.bn2d22c(x) for x in outs] 
        outs = [complex_tanh(x+y) for x,y in zip(outs,outs2)] 
        outs3 = outs

        out_max = torch.max(torch.stack([torch.abs(out) for out in outs],0),0)[0] 
        outmax = [out * (out_max /  (torch.abs(out)+1e-6)) for out in outs] 
 
        outs = [torch.cat((outs[idx], outmax[idx], xs[idx]),1) for idx in range(self.opts.stack_imgs)] 
 
        outs = [self.conv3a(x) for x in outs] 
        outs = [self.bn2d3a(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.conv3b(x) for x in outs] 
        outs = [self.bn2d3b(x) for x in outs] 
        outs = [complex_tanh(x+y) for x,y in zip(outs,outs3)] 
        outs = [self.conv3c(x) for x in outs] 
        outs = [self.bn2d3c(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
 
        ''' 
        outs = [x.transpose(1, 2) for x in outs] 
        outs = [x.reshape(x.size(0), x.size(1), -1) for x in outs] 
        outs = [self.fc1(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
        outs = [self.fc2(x) for x in outs] 
        outs = [complex_tanh(x) for x in outs] 
 
        #Final 
        outs = [x.view(x.size(0), x.size(1), 1, -1) for x in outs] 
        outs = [x.transpose(1, 2).transpose(2, 3).contiguous() for x in outs]''' 
 
        outs = [torch.stack(( x.real, x.imag), 1).squeeze() for x in outs]
        return outs

