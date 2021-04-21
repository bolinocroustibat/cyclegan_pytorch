#!/usr/bin/env python
# coding: utf-8

# # CycleGAN in Pytorch

# This youtube video from "Two minutes paper" provides a good summary of the method we are going to implement in this notebook:  
# 
# [![AI Learns to Synthesize Pictures of Animals | Two Minute Papers #152](https://img.youtube.com/vi/D4C1dB9UheQ/0.jpg)](https://www.youtube.com/embed/D4C1dB9UheQ)
# 
# I also recommend you to take a look at the project page by its authors: https://junyanz.github.io/CycleGAN/

# ## Hyperparameters (on top for convinience)

# In[ ]:


epoch = 0  # epoch to start training from
n_epochs = 10  # number of epochs of training
dataset_name = "horse2zebra"  # name of the dataset
batch_size = 1  # size of the batches
lr = 0.0002  # adam: learning rate
b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
decay_epoch = 1  # epoch from which to start lr decay
n_cpu = 2  # number of cpu threads to use during batch generation
img_height = 256  # size of image height
img_width = 256  # size of image width
channels = 3  # number of image channels
sample_interval = 100  # interval between saving generator outputs
checkpoint_interval = 1  # interval between saving model checkpoints
n_residual_blocks = 9  # number of residual blocks in generator
lambda_cyc = 10.0  # cycle loss weight
lambda_id = 5.0  # identity loss weight


# ## Imports

# In[ ]:


import datetime
import glob
import itertools
import math
import os
import random
import sys
import time
import zipfile

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from tqdm import trange
from tqdm.notebook import tqdm


# In[ ]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


if torch.cuda.is_available():
    print("This notebook will run on GPU.")
    device = "cuda"
else:
    print("This notebook will run on CPU.")
    device = "cpu"


# ## Utilities

# In[ ]:


# some helper functions to download the dataset
# this code comes mainly from gluoncv.utils
def download(url, path=None, overwrite=False) -> str:
    """Download an given URL.
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print("Downloading %s from %s..." % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get("content-length")
        with open(fname, "wb") as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit="KB",
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)
    return fname


def download_dataset(
    dataset_name: str, data_path: str = "data/", overwrite: bool = False
) -> None:
    compatible_datasets = [
        "ae_photos",
        "apple2orange",
        "cezanne2photo",
        "cityscapes",
        "facades",
        "grumpifycat",
        "horse2zebra",
        "iphone2dslr_flower",
        "maps",
        "mini",
        "mini_colorization",
        "mini_colorization",
        "mini_pix2pix",
        "monet2photo",
        "summer2winter_yosemi",
        "ukiyoe2photo",
        "vangogh2photo",
    ]
    if dataset_name not in compatible_datasets:
        print("The dataset you chose is not compatible.")
        print(f"Please select one among: {compatible_datasets}")
        return

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    download_url = f"https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset_name}.zip"
    download_dir = os.path.join(data_path, "downloads")
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    filename = download(download_url, path=download_dir, overwrite=overwrite)

    # Extract archive in target dir
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(path=data_path)

    # Re-organize dirs for more clarity
    testdir = data_path + dataset_name + "/" + "test"
    traindir = data_path + dataset_name + "/" + "train"
    if not os.path.exists(testdir):
        os.mkdir(testdir)
    if not os.path.exists(traindir):
        os.mkdir(traindir)
    try:
      os.rename(data_path + dataset_name + "/" + "trainA", traindir + "/A")
      os.rename(data_path + dataset_name + "/" + "trainB", traindir + "/B")
      os.rename(data_path + dataset_name + "/" + "testA", testdir + "/A")
      os.rename(data_path + dataset_name + "/" + "testB", testdir + "/B")
    except OSError:
      pass

    # Done
    print(f"Dataset downloaded and extracted in '{data_path}'.")


# In[ ]:


download_dataset(dataset_name, overwrite=False)


# ## Dataset

# In[ ]:


class ImagesData(Dataset):
    def __init__(self, root, data_augmentations=None, unaligned=False, dataset="train"):
        self.data_augmentations = data_augmentations
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{dataset}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{dataset}/B") + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Some images are graycale
        # So we need to convert them to RGB
        if image_A.mode != "RGB":
          image_A = self.to_rgb(image_A)
        if image_B.mode != "RGB":
          image_B = self.to_rgb(image_B)

        item_A = self.data_augmentations(image_A)
        item_B = self.data_augmentations(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    @staticmethod
    def to_rgb(image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image


# In[ ]:


# Normalization values are the ones from ImageNet
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# Training Data augmentations
train_data_augmentations = transforms.Compose(
    [
        transforms.Resize(int(img_height * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Test Data augmentations
test_data_augmentations = transforms.Compose(
    [
        transforms.Resize(int(img_height), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# In[ ]:


# Training Data Loader
train_loader = DataLoader(
    ImagesData(
        f"data/{dataset_name}",
        data_augmentations=train_data_augmentations,
        unaligned=True,
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)
# Test Data Loader
test_loader = DataLoader(
    ImagesData(
        f"data/{dataset_name}",
        data_augmentations=test_data_augmentations,
        unaligned=True,
        dataset="test",
    ),
    batch_size=5,
    shuffle=False,
    num_workers=1,
)


# ## Models

# ### Generator
# 
# From the paper, Appendix, 7.2. Network architectures: 
# 
# "We  adopt  our  architectures from  Johnson  et  al. [23].  
# We  use 6 residual  blocks  for 128×128 training images,  
# and 9 residual blocks for 256×256 or higher-resolution  
# training images. Below, we followthe naming convention  
# used in the Johnson et al.’s Github repository.  
# Let c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer  
# with k filters and stride 1. dk denotes a 3×3 Convolution-InstanceNorm-ReLU   
# layer with k filters and stride 2. Reflection padding was  
# used to reduce artifacts. Rk denotes a residual block that  
# contains two 3×3 convolutional layers with the same number  
# of filters on both layer. uk denotes a 3×3 fractional-strided-Convolution-InstanceNorm-ReLU  
# layer with k filters and stride 1/2. 
# 
# The network with 6 residual blocks consists of:  
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3  
# The network with 9 residual blocks consists of:  
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128u64,c7s1-3"  

# In[ ]:


# Let's implement first the residual block.
# We will re-use it many times
# in the code.


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


# In[ ]:


# Now let's implement the generator according
# to the paper.


class Generator(nn.Module):
    def __init__(self, input_shape: list, num_residual_blocks: int = 9):
        super(Generator, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        # c7s1-64
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        # d128, d256
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        # R256 * <num_residual_blocks> (6 or 9 in the paper)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        # u128, # u64
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # c7s1-3
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ### Discriminator
# 
# From the paper, Appendix, 7.2. Network architectures:
# 
# For discriminator networks, we use 70×70 PatchGAN [22].  
# Let Ck denote a 4×4 Convolution-InstanceNorm-LeakyReLU  
# layer with k filters and stride 2. After the last layer,  
# we apply a convolution to produce a1-dimensional output.  
# We do not use InstanceNorm for the first C64 layer.  
# We use leaky ReLUs with a slope of 0.2.  
# 
# The discriminator architecture is:  
# C64-C128-C256-C512  

# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, input_shape: list):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            # C64
            *discriminator_block(64, 128),
            # C128
            *discriminator_block(128, 256),
            # C256
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            # C512
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# In[ ]:


# Create sample and checkpoint directories
os.makedirs(f"images/{dataset_name}", exist_ok=True)
os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)


# In[ ]:


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

criterion_GAN.to(device)
criterion_cycle.to(device)
criterion_identity.to(device)


# ## Initializing our models

# In[ ]:


input_shape = (channels, img_height, img_width)


# In[ ]:


# Initialize generator and discriminator
G_AB = Generator(input_shape, n_residual_blocks)
G_BA = Generator(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)


# In[ ]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# In[ ]:


# Initialize weights
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)


# In[ ]:


G_AB.to(device)
G_BA.to(device)
D_A.to(device)
D_B.to(device)


# G_AB.to(device)
# G_BA.to(device)
# D_A.to(device)
# D_B.to(device)

# ## Optimizers

# In[ ]:


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))


# ## Learning rate scheduler

# In[ ]:


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


# In[ ]:


# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)


# In[ ]:


if device == "cuda":
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.Tensor


# From the paper, 4. Implemention, Training details: 
# 
# [...] to  reduce  model  oscillation  [15],   
# we  follow Shrivastava  et  al.’s  strategy  [46]   
# and  update  the  discriminators using a history  
# of generated images rather than the ones produced  
# by the latest generators.  We keep an image buffer  
# that stores the 50 previously created images.  

# In[ ]:


class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# In[ ]:


# Buffers of previously generated samples
fake_A_buffer = ImageBuffer()
fake_B_buffer = ImageBuffer()


# In[ ]:


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_loader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    writer.add_image("Sample", image_grid, batches_done)
    save_image(image_grid, f"images/{dataset_name}/{batches_done}.png", normalize=False)


# ## Training loop

# In[ ]:


# get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[ ]:


if device == "cuda":
    scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter()

writer.add_scalar("LearningRate/Generator", lr_scheduler_G.get_last_lr()[0], epoch)
writer.add_scalar(
    "LearningRate/DiscriminatorA", lr_scheduler_D_A.get_last_lr()[0], epoch
)
writer.add_scalar(
    "LearningRate/DiscriminatorB", lr_scheduler_D_B.get_last_lr()[0], epoch
)


for epoch in trange(epoch, n_epochs):
    pbar = tqdm(total=len(train_loader))
    for i, batch in enumerate(train_loader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False
        )
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False
        )

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        if device == "cuda":
            with autocast():
                # Identity loss
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
        else:
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        if device == "cuda":
            with autocast():
                # GAN loss
                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        else:
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        if device == "cuda":
            with autocast():
                # Cycle loss
                recov_A = G_BA(fake_B)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
        else:
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        if device == "cuda":
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

        else:
            loss_G.backward()
            optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        if device == "cuda":
            with autocast():
                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        else:
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        if device == "cuda":
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
        else:
            loss_D_A.backward()
            optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        if device == "cuda":
            with autocast():
                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        else:
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        if device == "cuda":
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
        else:
            loss_D_B.backward()
            optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        if device == "cuda":
            scaler.update()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i

        writer.add_scalar("Loss/Generators", loss_G, batches_done)
        writer.add_scalar("Loss/Discriminators", loss_D, batches_done)

        # Save predictions every 'sample_interval'
        if batches_done % sample_interval == 0:
            sample_images(batches_done)
        pbar.update(1)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    writer.add_scalar("LearningRate/Generator", lr_scheduler_G.get_last_lr()[0], epoch)
    writer.add_scalar(
        "LearningRate/DiscriminatorA", lr_scheduler_D_A.get_last_lr()[0], epoch
    )
    writer.add_scalar(
        "LearningRate/DiscriminatorB", lr_scheduler_D_B.get_last_lr()[0], epoch
    )

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), f"saved_models/{dataset_name}/G_AB_{epoch}.pth")
        torch.save(G_BA.state_dict(), f"saved_models/{dataset_name}/G_BA_{epoch}.pth")
        torch.save(D_A.state_dict(), f"saved_models/{dataset_name}/D_A_{epoch}.pth")
        torch.save(D_B.state_dict(), f"saved_models/{dataset_name}/D_B_{epoch}.pth")

