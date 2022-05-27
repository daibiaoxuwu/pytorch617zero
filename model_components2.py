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
class classificationHybridModel2(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim_in=2, conv_dim_out=128, conv_dim_lstm=1024):
        super(classificationHybridModel2, self).__init__()

        self.out_size = 1
        self.conv1 = nn.Conv2d(2, 16, (3, 3), stride=(2, 2), padding=(1, 1))
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
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

        self.conv1f = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(2, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential(

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(258, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

        self.conv2f = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(258, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

        self.conv22 = nn.Sequential(

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(258, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

        self.conv22f = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(258, 64, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64))

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
            nn.BatchNorm2d(8))

        self.conv3f = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(258, 8, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(8))

        #self.lstm = nn.LSTM( opts.conv_dim_lstm, opts.lstm_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(opts.conv_dim_lstm, opts.fc1_dim)
        self.fc2 = nn.Linear(opts.fc1_dim, opts.freq_size)

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
        outs = []
        xsnew = [x.transpose(2, 3).contiguous() for x in xs]
        for x in xsnew:
            out1 = self.conv1(x)
            out2 = self.conv1f(x)
            outs.append(nn.ReLU()(out1+out2))
        #self.save_samples(outs, self.opts, 1)
            

        outss = torch.stack(outs,1)
        outmax = torch.max(outss,1)[0]
        outmin = torch.min(outss,1)[0]
        outavg = torch.mean(outss,1)
        outs2 = []
        for idx in range(len(xs)):
            x = torch.cat((outs[idx],outmax,outmin,outavg,xsnew[idx]),1)
            out1 = self.conv2(x)
            out2 = self.conv2f(x)
            outs2.append( nn.ReLU()(out1+out2))
        #self.save_samples([outmax,], self.opts, 5)
        #self.save_samples([outmin,], self.opts, 6)
        #self.save_samples([outavg,], self.opts, 7)
        outss2 = torch.stack(outs2,1)
        outmax2 = torch.max(outss2,1)[0]
        outmin2 = torch.min(outss2,1)[0]
        outavg2 = torch.mean(outss2,1)
        outs3 = []
        for idx in range(len(xs)):
            x = torch.cat((outs2[idx],outmax2,outmin2,outavg2,xsnew[idx]),1)
            out21 = self.conv22(x)
            out22 = self.conv22f(x)
            outs3.append( nn.ReLU()(out21+out22))

        outss3 = torch.stack(outs3,1)
        outmax3 = torch.max(outss3,1)[0]
        outmin3 = torch.min(outss3,1)[0]
        outavg3 = torch.mean(outss3,1)
        outs4 = []
        for idx in range(len(xs)):
            x = torch.cat((outs2[idx],outmax3,outmin3,outavg3,xsnew[idx]),1)
            out31 = self.conv3(x)
            out32 = self.conv3f(x)
            outs4.append( nn.ReLU()(out31+out32))
        #self.save_samples([outmax,], self.opts, 8)
        #self.save_samples([outmin,], self.opts, 9)
        #self.save_samples([outavg,], self.opts, 10)

        for idx in range(len(xs)):
        #    outs[idx] = out
        #self.save_samples(outs, self.opts, 3)
        #for idx in range(len(xs)):
            out = outs4[idx].transpose(1, 2).contiguous()
            out = out.view(out.size(0), out.size(1), -1)
            #out, _ = self.lstm(out)
            #out = F.relu(out)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)

            out = out.view(out.size(0), out.size(1), 1, -1)
            out = out.transpose(1, 2).contiguous()
            out = out.transpose(2, 3).contiguous()
            outs[idx] = out * xs[idx]
        return outs
