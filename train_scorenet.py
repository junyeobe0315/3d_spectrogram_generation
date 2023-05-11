from torch.utils.data import DataLoader, Dataset
import torch
from collections import OrderedDict

from utils.load_data import *
import numpy as np
from scipy import signal

import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np

import torch
import functools
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import scipy
from models import *


def make_train_dataset(train_sub, y_num):
    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    test_sub=[],
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    ys = []
    for idx, x in enumerate(train_x):
        if train_y[idx] == y_num:
            ys.append(train_y[idx])
    train = []
    for idx, x in enumerate(train_x):
        f, t, stft = scipy.signal.stft(x,fs=250, nperseg=250)
        data = np.concatenate((stft.real, stft.imag), axis=0)
        train.append(data)
    return train, ys


class Stft_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), self.y[idx]
    
def train_scorenet(train_x, train_y):
    
    train_stft = Stft_dataset(train_x, train_y)

    beta_1 = 1e-4
    beta_T = 0.02
    T = 500
    shape = (44, 126, 16)
    device = torch.device('cuda')

    score_model = nn.DataParallel(Model(device, beta_1, beta_T, T))
    score_model = score_model.to(device)
    process = DiffusionProcess(beta_1, beta_T, T, score_model, device, shape)
    optim = torch.optim.Adam(score_model.parameters(), lr = 0.0001)

    total_iteration = 100
    current_iteration = 0

    train_stft = Stft_dataset(train_x, train_y)

    dataloader = torch.utils.data.DataLoader(train_stft, batch_size = 32, drop_last = True, num_workers = 0)
    dataiterator = iter(dataloader)

    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(total_iteration, [losses], prefix='Iteration ')
    pbar = tqdm(range(total_iteration))
    
    for epoch in pbar:
        losses = []
        num_items = 0
        
        for x, y in dataloader:
            data = x.to(device = device)
            loss = loss_fn(score_model, x, idx=None)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            losses.append(loss.item())
            num_items += x.shape[0]
        pbar.set_description("Average loss : {}".format(sum(losses) / num_items))
    return score_model