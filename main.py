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
from sklearn.preprocessing import StandardScaler


import tensorflow as tf

from braindecode.datasets import create_from_X_y
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet, EEGITNet, EEGNetv4
from braindecode.augmentation import AugmentedDataLoader
import scipy

from atcnet import *

def make_classifier_dataset(train_sub, val_sub, test_sub):
    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    valid_sub=val_sub,
    test_sub=test_sub,
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, val_x, val_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    
    for channel in range(22):
        scaler = StandardScaler()
        scaler.fit(train_x[:,channel,:])
        train_x[:,channel,:] = scaler.transform(train_x[:,channel,:])
        val_x[:,channel,:] = scaler.transform(val_x[:,channel,:])
        test_x[:,channel,:] = scaler.transform(test_x[:,channel,:])

    return train_x, train_y, val_x, val_y, test_x, test_y

def sample(sampler, model, num_images=32):
    model.eval()
    with torch.no_grad():
        images = []
        for i in tqdm(range(int(num_images))):
            x_T = torch.randn((1, 44, 126, 16))
            batch_images = sampler(x_T.cuda()).cpu()
            images.append((batch_images + 1) / 2)
        samples = torch.cat(images, dim=0).numpy()
    return samples

def train_scorenet_by_label(train_sub, device):
    train_x0, train_y0 = make_train_dataset(train_sub, 0)
    score_model0 = train_scorenet(train_x0, train_y0, device)

    train_x1, train_y1 = make_train_dataset(train_sub, 1)
    score_model1 = train_scorenet(train_x1, train_y1, device)

    train_x2, train_y2 = make_train_dataset(train_sub, 2)
    score_model2 = train_scorenet(train_x2, train_y2, device)

    train_x3, train_y3 = make_train_dataset(train_sub, 3)
    score_model3 = train_scorenet(train_x3, train_y3, device)
    return score_model0, score_model1, score_model2, score_model3

def augment(device, train_sub, model0, model1, model2, model3, batch_size=32):
    
    sampler0 = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model0, device=device, shape=(44, 126, 16))
    samples0 = sampler0.sampling(sampling_number=batch_size, only_final=True)
    
    sampler1 = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model1, device=device, shape=(44, 126, 16))
    samples1 = sampler1.sampling(sampling_number=batch_size, only_final=True)
    
    sampler2 = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model2, device=device, shape=(44, 126, 16))
    samples2 = sampler2.sampling(sampling_number=batch_size, only_final=True)

    sampler3 = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model3, device=device, shape=(44, 126, 16))
    samples3 = sampler3.sampling(sampling_number=batch_size, only_final=True)

    generated_signal0 = return_to_signal(samples0)
    generated_signal0y = [0.0 for i in range(len(generated_signal0))]
    
    generated_signal1 = return_to_signal(samples1)
    generated_signal1y = [1.0 for i in range(len(generated_signal1))]
    
    generated_signal2 = return_to_signal(samples2)
    generated_signal2y = [2.0 for i in range(len(generated_signal2))]
    
    generated_signal3 = return_to_signal(samples3)
    generated_signal3y = [3.0 for i in range(len(generated_signal3))]
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

    normalizer = StandardScaler()
    for idx, x in enumerate(train_x):
        normalizer.fit(x)
        train_x[idx] = normalizer.transform(x)
        
    
    train_y.extend(generated_signal0y)
    train_y.extend(generated_signal1y)
    train_y.extend(generated_signal2y)
    train_y.extend(generated_signal3y)
    
    return train_x, train_y


def return_to_signal(sample):
    generated_signal = []
    for batch in range(sample.shape[0]): # batch size
        sliced_sample = sample[batch]
        generated_stft_real = sliced_sample[:22].cpu() # real stft channel
        generated_stft_imag = sliced_sample[22:].cpu() # imaginary stft channel
        for idx, imag in enumerate(generated_stft_imag): 
            generated_stft_imag[idx] = np.multiply(imag, complex(0,1))
        generated_stft = np.add(generated_stft_real, generated_stft_imag)
        t, sig = scipy.signal.istft(generated_stft, fs=250, nperseg=250)
        generated_signal.append(sig)
    generated_signal = np.array(generated_signal)
    print("Generated signal shape : ", generated_signal.shape)
    return generated_signal


class ATC_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        y = torch.eye(4)
        return np.array(self.x[idx][:,375:1500]), np.array(y[int(self.y[idx])])

    def get_dataset(self):
        X = []
        Y = []
        for i in range(ATC_dataset.__len__(self)):
            x, y = ATC_dataset.__getitem__(self, i)
            x = x.reshape(1,22,1125)
            y = y.reshape(4)
            X.append(x)
            Y.append(y)
        return np.array(X).reshape(-1, 1, 22, 1125), np.array(Y).reshape(-1, 4)

def train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=32):
    train_x, train_y = augment(device, train_sub, score_model0, score_model1, score_model2, score_model3, batch_size=batch_size)
    
    train_set = ATC_dataset(train_x, train_y)

    _, _, valid_x, valid_y, test_x, test_y = make_classifier_dataset(train_sub, val_sub, test_sub)

    valid_set = ATC_dataset(valid_x, valid_y)
    test_set = ATC_dataset(test_x, test_y)

    X_train, y_train_onehot = train_set.get_dataset()
    X_test, y_test_onehot = valid_set.get_dataset()

    train_atcnet(X_train, y_train_onehot, X_test, y_test_onehot)

def main(train_sub, val_sub, test_sub, device):
    score_model0, score_model1, score_model2, score_model3 = train_scorenet_by_label(train_sub, device)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=32)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=64)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=128)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=256)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=512)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=1024)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=2048)


def train_without_aug(train_sub, val_sub, test_sub):
    train_x, train_y, valid_x, valid_y, _,_ = make_classifier_dataset(train_sub, val_sub, test_sub)

    train_set = ATC_dataset(train_x, train_y)
    valid_set = ATC_dataset(valid_x, valid_y)

    X_train, y_train_onehot = train_set.get_dataset()
    X_test, y_test_onehot = valid_set.get_dataset()

    train_atcnet(X_train, y_train_onehot, X_test, y_test_onehot)

if __name__ == "__main__":

    device = torch.device("cuda")

    if device == torch.device("cuda:0"):
        train_sub = [1,2,3,4,5,6,7,8]
        val_sub = [9]
        test_sub = [9]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)


        train_sub = [1,2,3,4,5,6,7,9]
        val_sub = [8]
        test_sub = [8]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)

    if device == torch.device("cuda:0"):
        train_sub = [1,2,3,4,5,6,7,8]
        val_sub = [9]
        test_sub = [9]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)


        train_sub = [1,2,3,4,5,6,7,9]
        val_sub = [8]
        test_sub = [8]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)

    if device == torch.device("cuda:1"):
        train_sub = [1,2,3,4,5,6,8,9]
        val_sub = [7]
        test_sub = [7]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)


        train_sub = [1,2,3,4,5,7,8,9]
        val_sub = [6]
        test_sub = [6]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)

    if device == torch.device("cuda:2"):
        train_sub = [1,2,3,4,6,7,8,9]
        val_sub = [5]
        test_sub = [5]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)


        train_sub = [1,2,3,5,6,7,8,9]
        val_sub = [4]
        test_sub = [4]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)

    if device == torch.device("cuda:3"):
        train_sub = [1,2,4,5,6,7,8,9]
        val_sub = [3]
        test_sub = [3]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)


        train_sub = [1,3,4,5,6,7,8,9]
        val_sub = [2]
        test_sub = [2]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub)



