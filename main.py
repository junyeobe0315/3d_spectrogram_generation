from train_scorenet import *
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler

from typing import Callable

from torch.utils.data import DataLoader, Dataset
import torch

from utils.load_data import *
import numpy as np
from scipy import signal

import os
import numpy as np
import torch
import torchvision
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer


from braindecode.datasets import create_from_X_y
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.augmentation import AugmentedDataLoader
from samplers import *
import scipy

def make_train_dataset(train_sub, y_num):
    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    test_sub=[],
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    train = []
    ys = []
    for idx, x in enumerate(train_x):
        if train_y[idx] == y_num:
            train.append(x)
            ys.append(train_y[idx])

    normalizer = Normalizer()
    for idx, x in enumerate(train_x):
        normalizer.fit(x)
        train_x[idx] = normalizer.transform(x)
            
    print("train length : ", len(train))
    return train, ys

def make_all_train_dataset(train_sub):
    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    test_sub=[],
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    normalizer = Normalizer()
    for idx, x in enumerate(train_x):
        normalizer.fit(x)
        train_x[idx] = normalizer.transform(x)
    return train_x, train_y

def make_valid_test_dataset(val_sub, test_sub):
    BCIC_dataset = load_BCIC(
    train_sub=[],
    valid_sub=val_sub,
    test_sub=test_sub,
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, valid_x, valid_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    normalizer = Normalizer()
    for idx, x in enumerate(valid_x):
        normalizer.fit(x)
        valid_x[idx] = normalizer.transform(x)
    
    normalizer = Normalizer()
    for idx, x in enumerate(test_x):
        normalizer.fit(x)
        test_x[idx] = normalizer.transform(x)
    
    return valid_x, valid_y, test_x, test_y

def sampling(score_model, sample_batch_size):
    device = 'cuda'
    sigma=0.1
    #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    ## Generate samples using the specified sampler.
    samples = ode_sampler(score_model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                batch_size=sample_batch_size, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3)
    return samples

def stft_to_signal(stft):
    print(stft.shape)
    t, x = scipy.signal.istft(stft, fs=250)
    return x

def train_scorenet_by_label(train_sub):
    train_x0, train_y0 = make_train_dataset(train_sub, 0)
    score_model0 = train_scorenet(train_x0, train_y0)

    train_x1, train_y1 = make_train_dataset(train_sub, 1)
    score_model1 = train_scorenet(train_x1, train_y1)

    train_x2, train_y2 = make_train_dataset(train_sub, 2)
    score_model2 = train_scorenet(train_x2, train_y2)

    train_x3, train_y3 = make_train_dataset(train_sub, 3)
    score_model3 = train_scorenet(train_x3, train_y3)
    return score_model0, score_model1, score_model2, score_model3

def augment(train_sub, score_model0, score_model1, score_model2, score_model3, batch_size=32):
    samples0 = sampling(score_model0, sample_batch_size=batch_size)
    samples1 = sampling(score_model1, sample_batch_size=batch_size)
    samples2 = sampling(score_model2, sample_batch_size=batch_size)
    samples3 = sampling(score_model3, sample_batch_size=batch_size)

    generated_signal0 = return_to_signal(samples0)
    generated_signal0y = [0. for i in range(len(generated_signal0))]
    
    generated_signal1 = return_to_signal(samples1)
    generated_signal1y = [1. for i in range(len(generated_signal1))]
    
    generated_signal2 = return_to_signal(samples2)
    generated_signal2y = [2. for i in range(len(generated_signal2))]
    
    generated_signal3 = return_to_signal(samples3)
    generated_signal3y = [3. for i in range(len(generated_signal3))]
    print("generated signal length : ", len(generated_signal0)*4)

    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    test_sub=[],
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, _, _ = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    train_x = train_x.tolist()
    train_y = train_y.tolist()

    train_x.extend(generated_signal0)
    train_x.extend(generated_signal1)
    train_x.extend(generated_signal2)
    train_x.extend(generated_signal3)
    
    train_y.extend(generated_signal0y)
    train_y.extend(generated_signal1y)
    train_y.extend(generated_signal2y)
    train_y.extend(generated_signal3y)
    
    temp = list(zip(train_x, train_y))
    random.shuffle(temp)
    train_x, train_y = zip(*temp)

    return train_x, train_y


def return_to_signal(sample):
    generated_signal = []
    for i in range(sample.shape[0]): # batch size 
        sliced_sample = sample[i][0]
        temp = []
        for j in range(sliced_sample.shape[0]):
            generated_stft = sliced_sample[j]
            t, sig = scipy.signal.istft(generated_stft.cpu(), fs=250)
            temp.append(sig[:1875])
        generated_signal.append(temp)
    return generated_signal


class stft_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx])

def train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=32):
    train_x, train_y = augment(train_sub, score_model0, score_model1, score_model2, score_model3, batch_size=batch_size)
    
    train_set = stft_dataset(train_x, train_y)

    valid_x, valid_y, test_x, test_y = make_valid_test_dataset(val_sub, test_sub)

    valid_set = stft_dataset(valid_x, valid_y)
    test_set = stft_dataset(test_x, test_y)

    n_classes = 4
    print(train_set.__getitem__(0)[0].shape)
    n_channels = train_set.__getitem__(0)[0].shape[0]
    input_window_samples = train_set.__getitem__(0)[0].shape[1]


    model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
    ).cuda()

    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 50

    clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    )

    clf.fit(train_set, y=None, epochs=n_epochs)

def main(train_sub, val_sub, test_sub):
    score_model0, score_model1, score_model2, score_model3 = train_scorenet_by_label(train_sub)
    train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=32)
    train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=64)
    train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=128)
    train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=256)
    train_with_aug(train_sub, val_sub, score_model0, score_model1, score_model2, score_model3, batch_size=512)

def train_witout_aug(train_sub, val_sub, test_sub):
    train_x, train_y = make_all_train_dataset(train_sub)
    valid_x, valid_y, test_x, test_y = make_valid_test_dataset(val_sub, test_sub)

    train_set = stft_dataset(train_x, train_y)
    valid_set = stft_dataset(valid_x, valid_y)
    test_set = stft_dataset(test_x, test_y)

    n_classes = 4
    print(train_set.__getitem__(0)[0].shape)
    n_channels = train_set.__getitem__(0)[0].shape[0]
    input_window_samples = train_set.__getitem__(0)[0].shape[1]

    model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
    ).cuda()

    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 50

    clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    )

    clf.fit(train_set, y=None, epochs=n_epochs)

if __name__ == "__main__":
    sigma = 0.1
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    device = "cuda"

    train_sub = [1,2,3,4,5,6,7,8]
    val_sub = [9]
    test_sub = [9]

    main(train_sub, val_sub, test_sub)
    train_witout_aug(train_sub, val_sub, test_sub)
