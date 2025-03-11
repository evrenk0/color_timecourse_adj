import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import csv
import os
import shutil


def get_blur(epoch, epochs_to_final=300, \
             start_sigma=5, final_sigma=0.125,\
             sigma_nonlin=False, base=None):
    """
    Parameters:
        epoch: what epoch of training are we on?
        epochs_to_final: how many epochs will it take to get to final blur value?
            this parameter adjusts how fast sigma will change over the course of training.
        start_sigma: initial value for blurring kernel size (pixels)
        final_sigma: final value for blurring kernel size (pixels)
            default is 1/8=0.125, because that results in kernel size=1 pixel.
        sigma_nonlin: should sigma change in a non-linear way?
            if False (default), then sigma will vary linearly.
        base: if sigma_nonlin=True, this specifies shape of the nonlinearity.
            Base>1 = faster change at beginning
            Base<1 = faster change at end

    """
    if epoch>=epochs_to_final:
        sigma = final_sigma
    else:
        if not sigma_nonlin:
            # assume sigma will change linearly over the training period.
            change_per_epoch = (start_sigma-final_sigma)/epochs_to_final
            sigma = start_sigma - (epoch * change_per_epoch)
        else:
            # sigma will change according to a logarithmic function with specified base.
            assert(base is not None and base!=1) # can't have base 1
            start_x_prime = base**(-(start_sigma-final_sigma))
            final_x_prime = 1;
            sigma_prime = epoch/epochs_to_final*(final_x_prime - start_x_prime) + start_x_prime
            sigma = (-1) * np.log(sigma_prime)/np.log(base) + final_sigma

    kernel_size = np.ceil(sigma*8)
    if not np.mod(kernel_size,2):
        kernel_size += 1

    return torchvision.transforms.GaussianBlur(kernel_size = kernel_size, sigma = sigma)
