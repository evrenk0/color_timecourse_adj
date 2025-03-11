import sys
print(sys.prefix)

import img_adj


# input_dir = 'C:/Users/evren/OneDrive/Desktop/test/for_evren/imges/input'  # input directory
# output_dir = 'C:/Users/evren/OneDrive/Desktop/test/for_evren/imges/output'  # output directory
# RG = 0.0 # red/green sensitivity parameter
# BY = 1.0 # blue/yellow sensitivity parameter

# # Process all images in input_dir, modify and store in ouput_dir
# img_adj.process_directory(input_dir, output_dir, RG, BY)

def run():
    input_dir = '/lab_data/hendersonlab/datasets/Ecoset'  # input directory
    output_dir = '/user_data/emkonuk/Scripts/imges/Ecoset_filters/RG05_BY00'  # output directory
    RG = 0.5 # red/green sensitivity parameter
    BY = 0 # blue/yellow sensitivity parameter

    # Process all images in input_dir, modify and store in ouput_dir
    img_adj.process_nested_directories(input_dir, output_dir, RG, BY)

    # # Process single level directory in input_dir, modify and sotre in output dir
    # img_adj.process_single_level_directory(input_dir, output_dir, RG, BY)





#############################################################################
#############################################################################
#############################################################################
# import os
# import numpy as np
# from PIL import Image
# from PIL import ImageStat
# import math

# import torch
# import torchvision   
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torchvision.datasets import ImageFolder

# # cuda = torch.cuda.is_available()
# # device = torch.device("cuda" if cuda else "cpu")
# # num_workers = 4 if cuda else 0

# import color_utils

# fn2load = 'C:/Users/evren/OneDrive/Desktop/test/for_evren/cabbage_08s.jpg'

# img_BW = Image.open(fn2load).convert('L')
# img_RGB = Image.open(fn2load)
# image = np.reshape(np.array(img_RGB.getdata()), [img_RGB.size[0],img_RGB.size[1],3])
# image_cielab = color_utils.rgb_to_CIELAB(image, device=None)


# # Upload image
# import cv2
# im_bgr = cv2.imread(fn2load)
# im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
# im_cielab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2Lab)


# # Subtract, centering color scales
# adjust_lab = np.array([1.0, 255/2, 255/2])
# adj_mat = np.tile(adjust_lab[None,None,:], [im_cielab.shape[0], im_cielab.shape[1],1])

# im_cielab = np.subtract(im_cielab, adj_mat)

# # Scaling color from 0.0 (B/W) - 1.0 (Original) for RG BY thresholds
# ##################################################
# adjust_lab = np.array([1.0, 0.5, 0.5])
# adj_mat = np.tile(adjust_lab[None,None,:], [im_cielab.shape[0], im_cielab.shape[1],1])

# im_cielab = np.multiply(im_cielab, adj_mat)
# ##################################################

# # Add values back in
# adjust_lab = np.array([1.0, 255/2, 255/2])
# adj_mat = np.tile(adjust_lab[None,None,:], [im_cielab.shape[0], im_cielab.shape[1],1])

# im_cielab = np.add(im_cielab, adj_mat)
# # Convert back to rgb space, save into im_rgb_adj
# im_rgb_adj = cv2.cvtColor(np.uint8(im_cielab), cv2.COLOR_Lab2RGB)




# # Save the image, first convert to pillow
# output_filename = "adjusted_image.png"
# im_rgb_adj_pil = Image.fromarray(np.uint8(im_rgb_adj))
# im_rgb_adj_pil.save(output_filename)


# # # Visualizing images
# # plt.figure()
# # plt.imshow(im_rgb_adj)
# # plt.show()

# # plt.figure()
# # plt.imshow(im_rgb)
# # plt.show()