import torch
import scipy
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from train_scorenet import *
from utils.load_data import *
from atcnet import *

from braindecode.datasets import create_from_X_y
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.models import ShallowFBCSPNet, EEGITNet, EEGNetv4
from braindecode.augmentation import AugmentedDataLoader
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler


def make_classifier_dataset(train_sub, val_sub, test_sub):
    BCIC_dataset = load_BCIC(
    train_sub=train_sub,
    valid_sub=val_sub,
    test_sub=test_sub,
    alg_name = 'Tensor_CSPNet',
    scenario = 'raw-signal-si'
    )

    train_x, train_y, val_x, val_y, test_x, test_y = BCIC_dataset.generate_training_valid_test_set_subject_independent()
    train_X = np.zeros(shape=(train_x.shape[0], train_x.shape[1], 1125))
    val_X = np.zeros(shape=(val_x.shape[0], val_x.shape[1], 1125))
    test_X = np.zeros(shape=(test_x.shape[0], test_x.shape[1], 1125))

    for channel in range(22):
        scaler = StandardScaler()
        scaler.fit(train_x[:,channel,:])
        train_x[:,channel,:] = scaler.transform(train_x[:,channel,:])
        val_x[:,channel,:] = scaler.transform(val_x[:,channel,:])
        test_x[:,channel,:] = scaler.transform(test_x[:,channel,:])
        train_X[:, channel] = train_x[:, channel, 375:1500]
        val_X[:, channel] = val_x[:, channel, 375:1500]
        test_X[:, channel] = test_x[:, channel, 375:1500]

    return train_X, train_y, val_X, val_y, test_X, test_y

def sample(sampler, label, device, num_images=32):
    label = torch.tensor(int(label), dtype=torch.int).to(device)
    with torch.no_grad():
        images = []
        for i in tqdm(range(int(num_images/32))):
            batch_images = sampler.sampling(sampling_number=32, label=label, only_final=True).cpu()
            images.append((batch_images + 1) / 2)
        samples = torch.cat(images, dim=0)
    return samples

def train_scorenet_by_label(train_sub, device):
    train_x, train_y = make_train_dataset(train_sub, [0.0])
    score_model0 = train_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_train_dataset(train_sub, [1.0])
    score_model1 = train_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_train_dataset(train_sub, [2.0])
    score_model2 = train_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_train_dataset(train_sub, [3.0])
    score_model3 = train_scorenet(train_x, train_y, device)
    return score_model0, score_model1, score_model2, score_model3

def train_raw_scorenet_by_label(train_sub, device):
    train_x, train_y = make_raw_train_dataset(train_sub, [0.0])
    score_model0 = train_raw_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_raw_train_dataset(train_sub, [1.0])
    score_model1 = train_raw_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_raw_train_dataset(train_sub, [2.0])
    score_model2 = train_raw_scorenet(train_x, train_y, device)
    
    train_x, train_y = make_raw_train_dataset(train_sub, [3.0])
    score_model3 = train_raw_scorenet(train_x, train_y, device)
    return score_model0, score_model1, score_model2, score_model3


def augment(device, train_sub, model0, model1, model2, model3, batch_size=32):
    
    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model0, device=device, shape=(44, 188, 4))
    samples0 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model1, device=device, shape=(44, 188, 4))
    samples1 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model2, device=device, shape=(44, 188, 4))
    samples2 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=500, diffusion_fn=model3, device=device, shape=(44, 188, 4))
    samples3 = sample(sampler, label=0, device=device, num_images=batch_size)

    generated_signal0 = return_to_signal(samples0)
    generated_signal0y = [0.0 for i in range(len(generated_signal0))]
    
    generated_signal1 = return_to_signal(samples1)
    generated_signal1y = [1.0 for i in range(len(generated_signal1))]
    
    generated_signal2 = return_to_signal(samples2)
    generated_signal2y = [2.0 for i in range(len(generated_signal2))]
    
    generated_signal3 = return_to_signal(samples3)
    generated_signal3y = [3.0 for i in range(len(generated_signal3))]
    print("generated signal length : ", len(generated_signal0)*4)

    train_x, train_y, _, _, _, _ = make_classifier_dataset(train_sub, val_sub=[1], test_sub=[1])
    
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

    return train_x, train_y

def raw_augment(device, train_sub, model0, model1, model2, model3, batch_size=32):
    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=1000, diffusion_fn=model0, device=device, shape=(44, 188, 4))
    samples0 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=1000, diffusion_fn=model1, device=device, shape=(44, 188, 4))
    samples1 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=1000, diffusion_fn=model2, device=device, shape=(44, 188, 4))
    samples2 = sample(sampler, label=0, device=device, num_images=batch_size)

    sampler = DiffusionProcess(beta_1=1e-4, beta_T=0.02, T=1000, diffusion_fn=model3, device=device, shape=(44, 188, 4))
    samples3 = sample(sampler, label=0, device=device, num_images=batch_size)


def return_to_signal(sample):
    generated_signal = []
    for batch in range(sample.shape[0]): # batch size
        sliced_sample = sample[batch]
        generated_stft_real = sliced_sample[:22] # real stft channel
        generated_stft_imag = sliced_sample[22:] # imaginary stft channel
        for idx, imag in enumerate(generated_stft_imag): 
            generated_stft_imag[idx] = np.multiply(imag, complex(0,1))
        generated_stft = np.add(generated_stft_real, generated_stft_imag)
        t, sig = scipy.signal.istft(generated_stft, fs=250, nperseg=375, noverlap=0)
        generated_signal.append(sig[:,:1125])
    generated_signal = np.array(generated_signal)
    for channel in range(22):
        scaler = StandardScaler()
        scaler.fit(generated_signal[:,channel,:])
        generated_signal[:,channel,:] = scaler.transform(generated_signal[:,channel,:])
    print("Generated signal shape : ", generated_signal.shape)
    return generated_signal


class ATC_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        
        return np.array(self.x[idx]), self.y[idx]

    def get_dataset(self):
        X = []
        Y = []
        for i in range(ATC_dataset.__len__(self)):
            x, y = ATC_dataset.__getitem__(self, i)
            x = x.reshape(1,22,-1)
            x = x[:,:,:1125]
            X.append(x)
            Y.append(y)
        return np.array(X).reshape(-1, 22, 1125), Y

def train_with_aug(device, train_sub, val_sub, test_sub, model0, model1, model2, model3, batch_size):
    train_x, train_y = augment(device, train_sub, model0, model1, model2, model3, batch_size=batch_size)
    train_set = ATC_dataset(train_x, train_y)
    _, _, valid_x, valid_y, test_x, test_y = make_classifier_dataset(train_sub, val_sub, test_sub)

    valid_set = ATC_dataset(valid_x, valid_y)
    valid_x, valid_y = valid_set.get_dataset()

    weight_decay = 0

    model = EEGNetv4(
    in_chans=22,
    n_classes=4,
    input_window_samples=1125,
    final_conv_length='auto'
    ).to(device)
    lr = 0.0625 * 0.01

    batch_size = 64
    n_epochs = 200

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

def train_raw_with_aug(device, train_sub, val_sub, test_sub, model0, model1, model2, model3, batch_size):
    train_x, train_y = raw_augment(device, train_sub, model0, model1, model2, model3, batch_size=batch_size)
    
    train_set = ATC_dataset(train_x, train_y)
    train_x, train_y = train_set.get_dataset()

    _, _, valid_x, valid_y, test_x, test_y = make_classifier_dataset(train_sub, val_sub, test_sub)

    valid_set = ATC_dataset(valid_x, valid_y)
    valid_x, valid_y = valid_set.get_dataset()

    weight_decay = 0
    model = EEGNetv4(
    in_chans=22,
    n_classes=4,
    input_window_samples=1125,
    final_conv_length='auto'
    ).to(device)
    lr = 0.0625 * 0.01

    batch_size = 64
    n_epochs = 200

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


def main(train_sub, val_sub, test_sub, device):
    score_model0, score_model1, score_model2, score_model3 = train_scorenet_by_label(train_sub, device)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=64)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=128)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=256)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=512)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=1024)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=2048)

def raw_main(train_sub, val_sub, test_sub, device):
    score_model0, score_model1, score_model2, score_model3 = train_raw_scorenet_by_label(train_sub, device)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=64)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=128)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=256)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=512)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=1024)
    train_with_aug(device, train_sub, val_sub, test_sub, score_model0, score_model1, score_model2, score_model3, batch_size=2048)



def train_without_aug(train_sub, val_sub, test_sub, device):
    train_x, train_y, valid_x, valid_y, _,_ = make_classifier_dataset(train_sub, val_sub, test_sub)

    train_set = ATC_dataset(train_x, train_y)
    train_x, train_y = train_set.get_dataset()

    _, _, valid_x, valid_y, test_x, test_y = make_classifier_dataset(train_sub, val_sub, test_sub)

    valid_set = ATC_dataset(valid_x, valid_y)
    valid_x, valid_y = valid_set.get_dataset()

    weight_decay = 0
    model = EEGNetv4(
    in_chans=22,
    n_classes=4,
    input_window_samples=1125,
    final_conv_length='auto'
    ).to(device)
    lr = 0.0625 * 0.01

    batch_size = 64
    n_epochs = 200

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

    device = torch.device("cuda")

    if device == torch.device("cuda:0"):
        train_sub = [1,2,3,4,5,6,7,8]
        val_sub = [9]
        test_sub = [9]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)


        train_sub = [1,2,3,4,5,6,7,9]
        val_sub = [8]
        test_sub = [8]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)

    if device == torch.device("cuda:1"):
        train_sub = [1,2,3,4,5,6,8,9]
        val_sub = [7]
        test_sub = [7]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)


        train_sub = [1,2,3,4,5,7,8,9]
        val_sub = [6]
        test_sub = [6]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)

    if device == torch.device("cuda:2"):
        train_sub = [1,2,3,4,6,7,8,9]
        val_sub = [5]
        test_sub = [5]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)


        train_sub = [1,2,3,5,6,7,8,9]
        val_sub = [4]
        test_sub = [4]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)

    if device == torch.device("cuda:3"):
        train_sub = [1,2,4,5,6,7,8,9]
        val_sub = [3]
        test_sub = [3]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)


        train_sub = [1,3,4,5,6,7,8,9]
        val_sub = [2]
        test_sub = [2]
        main(train_sub, val_sub, test_sub, device)
        train_without_aug(train_sub, val_sub, test_sub, device)



