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

import img_adj



import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import torch
import time



# Returns Function: path_to_img -> numpy img array
def get_img_adj(epoch, epochs_to_final=200,
                start_sigma = 0.0, final_sigma = 1.0):
    """
    Returns a version of the `img transform` with RG and BY values scaled linearly for the current epoch.

    Parameters:
        epoch (int): what epoch of training are we on?
        epochs_to_final (int): how many epochs will it take to get to final blur value?
            this parameter adjusts how fast sigma will change over the course of training.
        start_sigma (float): Starting sigma.
        final_sigma (float): Final sigma.

    Returns:
        function: A version of the `img transform` function with scaled RG and BY values.
    """
    # Set epoch to epochs_to_final if threshold has been exceeded
    if epoch >= epochs_to_final:
        epoch = epochs_to_final
    # Linearly interpolate sigma between start_sigma and final_sigma
    sigma = start_sigma + (final_sigma - start_sigma) * (epoch / epochs_to_final)

    # Scale RG and BY values using sigma
    RG = 1.0
    BY = sigma

    transform = img_adj.MyCustomTransform(RG, BY)
    def new_function(img):
        return transform(img)
    return new_function