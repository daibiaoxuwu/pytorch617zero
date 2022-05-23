"""Helpful functions for project."""
import os
import torch
from torch.autograd import Variable
from datetime import datetime

from random import shuffle
import numpy as np
import functools
import operator


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def spec_to_network_input(x, opts):
    """Converts numpy to variable."""
    freq_size = opts.freq_size
    # trim
    trim_size = freq_size // 2
    # up down 拼接
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    y_abs = torch.abs(y)
    y_abs_max = torch.tensor(
        list(map(lambda x: torch.max(x), y_abs)))
    y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
    y = torch.div(y, y_abs_max)

    if opts.x_image_channel == 2:
        y = torch.view_as_real(y)  # [B,H,W,2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)  # [B,H,W]
        y = torch.unsqueeze(y, 1)  # [B,H,W]
    return y  # [B,2,H,W]


def set_gpu(free_gpu_id):
    """Converts numpy to variable."""
    torch.cuda.set_device(free_gpu_id)


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Time =", current_time)
    
    strlist = [
    ' data_dir              '+str(opts.data_dir),
    ' snr_list              '+str(opts.snr_list),
    ' stack_imgs            '+str(opts.stack_imgs),
    ' batch_size            '+str(opts.batch_size),
    ' lr                    '+str(opts.lr),
    ' w_image               '+str(opts.w_image),
    ' checkpoint_dir        '+str(opts.checkpoint_dir),
    ' load_checkpoint_dir   '+str(opts.load_checkpoint_dir),
    ' load                  '+str(opts.load),
    ' load_iters            '+str(opts.load_iters)]
    print('\n'.join(strlist))
    print('=' * 80)
    return strlist
