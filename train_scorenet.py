from torch.utils.data import DataLoader, Dataset
import torch

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
from tqdm import tqdm

class Stft_datset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), self.y[idx]

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases 22 44 FFT  FFT time : 6
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(34,1,1), padding=(0,0,0), bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1,97,1), padding=(0,0,0), bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1,1,14), padding=(0,0,0), bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(4,10,2), padding=(0,0,0), bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.conv5 = torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(4,10,2), padding=(0,0,0), bias=False)
        self.dense9 = Dense(embed_dim, 512)
        self.gnorm5 = nn.GroupNorm(32, num_channels=512)
        self.conv6 = torch.nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(4,10,1), padding=(0,0,0), bias=False)
        self.dense10 = Dense(embed_dim, 1024)
        self.gnorm6 = nn.GroupNorm(32, num_channels=1024)

        # Decoding layers where the resolution increases
        self.tconv6 = torch.nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size= (4,10,1), padding=(0,0,0), bias=False)
        self.dense11 = Dense(embed_dim, 512)
        self.tgnorm6 = nn.GroupNorm(32, num_channels=512)

        self.tconv5 = torch.nn.ConvTranspose3d(in_channels=1024, out_channels=256, kernel_size= (4,10,2), padding=(0,0,0), bias=False)
        self.dense8 = Dense(embed_dim, 256)
        self.tgnorm5 = nn.GroupNorm(32, num_channels=256)
        self.tconv4 = torch.nn.ConvTranspose3d(in_channels=512, out_channels=128, kernel_size= (4,10,2), padding=(0,0,0), bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=64, kernel_size= (1,1,14), padding=(0,0,0), bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size= (1,97,1), padding=(0,0,0), bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size= (34,1,1), padding=(0,0,0), bias=False)


        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        # print(h1.shape)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        # print(h2.shape)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        # print(h3.shape)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # print(h4.shape)

        h5 = self.conv5(h4)
        h5 += self.dense9(embed)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)

        h6 = self.conv6(h5)
        h6 += self.dense10(embed)
        h6 = self.gnorm6(h6)
        h6 = self.act(h6)

        # Decoding path
        h = self.tconv6(h6)

        h += self.dense11(embed)
        h = self.tgnorm6(h)
        h = self.act(h)
        h = self.tconv5(torch.cat([h, h5], dim=1))

        h += self.dense8(embed)
        h = self.tgnorm5(h)
        h = self.act(h)
        h = self.tconv4(torch.cat([h, h4], dim=1))

        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None, None]
        return h
def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  

    Returns:
    The standard deviation.
    """    
    t = torch.tensor(t, device="cuda")
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device="cuda")
  
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1,2,3,4)))
    return loss
    
    
def train_scorenet(train_x, train_y, sigma):
    
    train_stft = Stft_datset(train_x, train_y)

    train_dataloader = DataLoader(train_stft, batch_size=32, num_workers=0)

    device = 'cuda' 


    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)



    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    n_epochs = 3000

    lr=1e-2


    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10, factor=0.9)

    tqdm_epoch = tqdm(range(n_epochs))
    losses = []

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in train_dataloader:
            x = x.to(device).unsqueeze(1)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        scheduler.step(avg_loss)
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Score model Average Loss: {:5f}, lr : {}'.format(avg_loss / num_items, optimizer.param_groups[0]['lr']))
        if (avg_loss / num_items < 100) or (optimizer.param_groups[0]['lr'] < 1e-7) :
            break
    return score_model