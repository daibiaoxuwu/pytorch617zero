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

def norm(y):
    for i in range(y.shape[1]):
        y_abs = torch.abs(y[:,i])
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y[:,i] = torch.div(y[:,i], y_abs_max)
    return y
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
        self.writeindex = 0

        self.conv1 = nn.Sequential(
            # cnn1
            nn.ZeroPad2d(2),
            nn.Conv2d(opts.x_image_channel, 256, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(256), nn.LeakyReLU(),

            )

        self.conv2 = []
        for i in range(2):
            self.conv2.append( nn.Sequential(

                # cnn3
                nn.Conv2d(256*3+2, 64, kernel_size=(1, 1), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.LeakyReLU(),

                nn.ZeroPad2d(1),
                nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.LeakyReLU(),

                nn.Conv2d(64, 256, kernel_size=(1, 1), dilation=(1, 1)),
                nn.BatchNorm2d(256)))
        self.conv2 = nn.ModuleList(self.conv2)


        self.conv3 = nn.Sequential(

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(256, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.LeakyReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.LeakyReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8))

        #self.lstm = nn.LSTM( opts.conv_dim_lstm, opts.lstm_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(opts.conv_dim_lstm , opts.fc1_dim)
        self.fc2 = nn.Linear(opts.fc1_dim, opts.freq_size * opts.out_channel)
        self.final = nn.Sigmoid()

    def merge_images(self, sources, opts):
        """Creates a grid consisting of pairs of columns, where the first column in
        each pair contains images source images and the second column in each pair
        contains images generated by the CycleGAN from the corresponding images in
        the first column.
        """
        b, c, h, w = sources[0].shape
        row = int(np.sqrt(c))
        column = math.ceil(c / row)
        merged = np.zeros([row * h * opts.stack_imgs, column * w])
        for stack_idx in range(len(sources)):
            for idx, s in enumerate(sources[stack_idx][0]):
                i = idx // column
                j = idx % column
                merged[(i * opts.stack_imgs + stack_idx) * h:(i * opts.stack_imgs + 1 + stack_idx) * h, j * w: (j + 1) * w] = s
        return merged.transpose(0,1)


    def save_samples(self, fixed_Y, opts, lvidx):
        """Saves samples from both generators X->Y and Y->X.
        """
        Y = [to_data(i) for i in fixed_Y]

        merged = self.merge_images(Y, opts)

        path = os.path.join(opts.checkpoint_dir, 'sample-Y'+str(lvidx)+'.png')
        merged = (merged - np.amin(merged)) / (np.amax(merged) - np.amin(merged)) * 255
        merged = cv2.flip(merged, 0)
        cv2.imwrite(path, merged)
        print('SAMPLE: {}'.format(path))

    def forward(self, xs):
        self.writeindex+=1
        xsnew = [x.transpose(2, 3) for x in xs]
        
        # CNN_1
        outs = [self.conv1(x) for x in xsnew]


        for i in range(2):

            # Merge
            outss = torch.stack(outs,1)
            outavg = torch.mean(outss,1)
            outmax = torch.max(outss,1)[0]-0.01*outavg
            outmin = torch.min(outss,1)[0]+0.01*outavg
            '''
            out_abs = torch.mean(torch.stack([torch.abs(out[:,0]+1j*out[:,1]) for out in outs],0),0)
            outavg = [out * (out_abs / torch.abs(out[:,0]+1j*out[:,1])).unsqueeze(1).repeat(1, 2, 1, 1) for out in outs]
            out_max = torch.max(torch.stack([torch.abs(out[:,0]+1j*out[:,1]) for out in outs],0),0)[0]
            outmax = [out * (out_abs / torch.abs(out[:,0]+1j*out[:,1])).unsqueeze(1).repeat(1, 2, 1, 1) for out in outs]-0.01*outavg
            out_min = torch.min(torch.stack([torch.abs(out[:,0]+1j*out[:,1]) for out in outs],0),0)[0]
            outmin = [out * (out_abs / torch.abs(out[:,0]+1j*out[:,1])).unsqueeze(1).repeat(1, 2, 1, 1) for out in outs]+0.01*outavg'''

            outs2 = []
            for idx in range(len(xs)):
                # Distribute
                x = torch.cat((outs[idx],outmax,outmin,xsnew[idx]),1)

                # CNN_2
                out1 = self.conv2[i](x)
                outs2.append( out1 + outs[idx] )
            outs = outs2

        for idx in range(len(xs)):
            out = self.conv3(outs[idx]).transpose(1, 2)
            out = out.reshape(out.size(0), out.size(1), -1)
            #out, _ = self.lstm(out)
            out = nn.LeakyReLU()(out)
            out = self.fc1(out)
            out = nn.LeakyReLU()(out)
            out = self.fc2(out)
            out = self.final(out)

            out = out.view(out.size(0), out.size(1), self.opts.out_channel, -1)
            out = out.transpose(1, 2)
            out = out.transpose(2, 3)
            #outs[idx] = (out + xs[idx]).contiguous() ##change
            outs[idx] = (out).contiguous()
        return outs
