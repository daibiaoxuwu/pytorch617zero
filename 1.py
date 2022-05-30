import os
import random
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import pickle
import sys


lora_img = np.array(scio.loadmat('test.mat')['chirp'].tolist())
lora_img = np.squeeze(lora_img)
lora_img = torch.tensor(lora_img, dtype=torch.cfloat)
print(lora_img.shape)
