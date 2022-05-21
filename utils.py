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
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Time =", current_time)

    print('--batch_size', opts.batch_size)
    print('--scaling_for_imaging_loss', opts.scaling_for_imaging_loss)
    print('--scaling_for_classification_loss', opts.scaling_for_classification_loss)
    print('--data_dir', opts.data_dir)
    print("--snr_list", opts.snr_list)
    print('--ratio_bt_train_and_test', opts.ratio_bt_train_and_test)
    print('--checkpoint_dir', opts.checkpoint_dir)
    print('--load', opts.load)
    print('--log_step', opts.log_step)
    print('--test_step', opts.test_step)
    print('--train_iters', opts.train_iters)
    print('--checkpoint_every', opts.checkpoint_every)
    print('--load_checkpoint_dir', opts.load_checkpoint_dir)
    print('--stack_imgs', opts.stack_imgs)

    print('=' * 80)
