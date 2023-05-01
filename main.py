from train_scorenet import *

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
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.plotting import get_grid
from lib.flows.elemwise import LogitTransform
from lib.helpers import logging, create
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader

from braindecode.datasets import create_from_X_y
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

    print("train length : ", len(train))
    return train, ys

def make_valid_test_dataset(val_sub, test_sub):
    BCIC_dataset = load_BCIC(
    train_sub=[],
    valid_sub=val_sub,
    test_sub=test_sub,
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, valid_x, valid_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
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

def augment(train_sub, score_model0, score_model1, score_model2, score_model3):
    samples0 = sampling(score_model0, sample_batch_size=32)
    samples1 = sampling(score_model1, sample_batch_size=32)
    samples2 = sampling(score_model2, sample_batch_size=32)
    samples3 = sampling(score_model3, sample_batch_size=32)

    for i in range(samples0.shape[0]):
        print(samples0.shape)



if __name__ == "__main__":
    sigma = 0.1
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    device = "cuda"

    train_sub = [1]
    val_sub = [2]
    test_sub = [3]

    score_model0, score_model1, score_model2, score_model3 = train_scorenet_by_label(train_sub)
    augment(train_sub, score_model0, score_model1, score_model2, score_model3)


