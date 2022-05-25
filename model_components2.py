# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
import numpy as np
import os
from utils import *

def norm(y):
    for i in range(y.shape[1]):
        y_abs = torch.abs(y[:,i])
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y[:,i] = torch.div(y[:,i], y_abs_max)
    return y
class classificationHybridModel2(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim_in=1, conv_dim_out=128, conv_dim_lstm=1024):
        super(classificationHybridModel2, self).__init__()

        self.out_size = 1
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(2, 2), padding=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dense = nn.Linear(conv_dim_lstm * 4, conv_dim_out * 4)
        self.fcn1 = nn.Linear(conv_dim_out * 4, conv_dim_out * 2)
        self.fcn2 = nn.Linear(2 * conv_dim_out, conv_dim_out)
        self.softmax = nn.Softmax(dim=1)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.act = nn.ReLU()

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


class maskCNNModel2(nn.Module):
    def __init__(self, opts):
        super(maskCNNModel2, self).__init__()
        self.opts = opts
        self.writeindex = 0

        self.conv1 = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(opts.x_image_channel, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU())

        self.conv2 = nn.Sequential(

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(258, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU())

        self.conv3 = nn.Sequential(

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(258, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU())


        #self.lstm = nn.LSTM( opts.conv_dim_lstm, opts.lstm_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(opts.conv_dim_lstm, opts.fc1_dim)
        self.fc2 = nn.Linear(opts.fc1_dim, opts.freq_size)

    def forward(self, xs):
        self.writeindex+=1
        outs = [0]*len(xs)
        xsnew = [x.transpose(2, 3).contiguous() for x in xs]
        for idx, x in enumerate(xsnew):
            outs[idx] = self.conv1(x)
            

        outss = torch.stack(outs,1)
        outmax = torch.max(outss,1)[0]
        outmin = torch.min(outss,1)[0]
        outavg = torch.mean(outss,1)
        for idx in range(len(xs)):
            outs[idx] = torch.cat((outs[idx],outmax,outmin,outavg,xsnew[idx]),1)

        for idx in range(len(xs)):
            outs[idx] = self.conv2(outs[idx])
        outss = torch.stack(outs,1)
        outmax = torch.max(outss,1)[0]
        outmin = torch.min(outss,1)[0]
        outavg = torch.mean(outss,1)
        for idx in range(len(xs)):
            outs[idx] = torch.cat((outs[idx],outmax,outmin,outavg,xsnew[idx]),1)

        for idx in range(len(xs)):
            out = self.conv3(outs[idx])
            out = out.transpose(1, 2).contiguous()
            out = out.view(out.size(0), out.size(1), -1)
            #out, _ = self.lstm(out)
            #out = F.relu(out)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)

            out = out.view(out.size(0), out.size(1), 1, -1)
            out = out.transpose(1, 2).contiguous()
            out = out.transpose(2, 3).contiguous()
            outs[idx] = out
        return outs
