import os, sys
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import color_utils

import numpy as np
from skimage import color
import torch
import time

# Process single image
def process_image(fn2load, RG, BY, output_dir):
    # Load and convert to RGB + CIE
    im_bgr = cv2.imread(fn2load)
    
    ##########################
    # Failsafe. If imread processes incorrectly due to corrupt images file, terminate function  
    # sys.out.flush() prints all statements in cache
    if im_bgr is None:
        print(f"Error: Could not load file {fn2load}")
        sys.stdout.flush()
        return
    if not os.path.exists(fn2load):
        print(f"Error: File does not exist {fn2load}")
        sys.stdout.flush()
        return
    if os.path.getsize(fn2load) == 0:
        print(f"Error: File is empty {fn2load}")
        sys.stdout.flush()
        return
    ##########################
    
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_cielab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2Lab)

    # Subtract, centering color scales
    adjust_lab = np.array([1.0, 255/2, 255/2])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.subtract(im_cielab, adj_mat)

    # Scaling color from 0.0 (B/W) - 1.0 (Original) for RG BY thresholds
    adjust_lab = np.array([1.0, RG, BY])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.multiply(im_cielab, adj_mat)

    # Add values back in
    adjust_lab = np.array([1.0, 255/2, 255/2])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.add(im_cielab, adj_mat)

    # Convert back to RGB space
    im_rgb_adj = cv2.cvtColor(np.uint8(im_cielab), cv2.COLOR_Lab2RGB)
    im_rgb_adj_pil = Image.fromarray(im_rgb_adj)

    # Resize and center the image to 224x224 without padding
    im_rgb_adj_pil = resize_and_crop_center(im_rgb_adj_pil, (224, 224))
    
    # Output file path
    output_filename = os.path.join(output_dir, os.path.basename(fn2load))
    im_rgb_adj_pil.save(output_filename)




# Function to resize and center-crop an image to specific dimensions
def resize_and_crop_center(image, target_size):
    # Get the original dimensions
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Determine the scaling factor (no padding, center crop only)
    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image while keeping the aspect ratio
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Calculate coordinates to crop the center of the image
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # Center-crop the image to target size
    image = image.crop((left, top, right, bottom))

    return image

# To process entire Ecoset folder
def process_nested_directories(input_dir, output_dir, RG, BY):
    for root, subdirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_filepath = os.path.join(root, filename)
                process_image(input_filepath, RG, BY, current_output_dir)

# To process just test, train, val
def process_single_level_directory(input_dir, output_dir, RG, BY):
    # Iterate over all subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):  # Only process directories within the input directory
            current_output_dir = os.path.join(output_dir, subdir)
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

            # Process files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
                    input_filepath = os.path.join(subdir_path, filename)
                    process_image(input_filepath, RG, BY, current_output_dir)





# # Function to obtain color adjustment function
# def get_process (epoch, epochs_to_final=1000, \
#              start_sigma=5, final_sigma=0.125,\
#              sigma_nonlin=False, base=None)


# Function that does color adjustment
def process (fn2load, RG, BY):
    
    # Load and convert to RGB + CIE
    im_bgr = cv2.imread(fn2load)
    
    ##########################
    # Failsafe. If imread processes incorrectly due to corrupt images file, terminate function  
    # sys.out.flush() prints all statements in cache
    if im_bgr is None:
        print(f"Error: Could not load file {fn2load}")
        sys.stdout.flush()
        return
    if not os.path.exists(fn2load):
        print(f"Error: File does not exist {fn2load}")
        sys.stdout.flush()
        return
    if os.path.getsize(fn2load) == 0:
        print(f"Error: File is empty {fn2load}")
        sys.stdout.flush()
        return
    ##########################
    
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_cielab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2Lab)


    device = 'cpu:0'    
    image_rgb_tensor = torch.Tensor(im_rgb.astype(np.single)).to(device)
    image_rgb_tensor = image_rgb_tensor[:,:,:,None]
    image_rgb_tensor = image_rgb_tensor/255.0       
    image_cielab_tensor = color_utils.RGB_to_CIELAB_tensor(image_rgb_tensor)

    
    im_cielab = image_cielab_tensor[:,:,:,0]

    # Subtract, centering color scales
    adjust_lab = np.array([1.0, 255/2, 255/2])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.subtract(im_cielab, adj_mat)

    # Scaling color from 0.0 (B/W) - 1.0 (Original) for RG BY thresholds
    adjust_lab = np.array([1.0, RG, BY])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.multiply(im_cielab, adj_mat)

    # Add values back in
    adjust_lab = np.array([1.0, 255/2, 255/2])
    adj_mat = np.tile(adjust_lab[None, None, :], [im_cielab.shape[0], im_cielab.shape[1], 1])
    im_cielab = np.add(im_cielab, adj_mat)


    image_cielab_tensor_adj = im_cielab[:,:,:,None].type(torch.float32)
    image_rgb_tensor_weighted = color_utils.CIELAB_to_RGB_tensor(image_cielab_tensor_adj)    
    image_rgb_weighted = image_rgb_tensor_weighted[:,:,:,0].detach().cpu().numpy() * 255.0
    image_rgb_weighted = np.minimum(np.maximum(image_rgb_weighted, 0), 255).astype('uint8')
    plt.imshow(image_rgb_weighted)

    # # Convert back to RGB space
    # im_rgb_adj = cv2.cvtColor(np.uint8(im_cielab), cv2.COLOR_Lab2RGB)
    # im_rgb_adj_pil = Image.fromarray(im_rgb_adj)

    # # Resize and center the image to 224x224 without padding
    # im_rgb_adj_pil = resize_and_crop_center(im_rgb_adj_pil, (224, 224))
    
    # # Output file path
    # output_filename = os.path.join(output_dir, os.path.basename(fn2load))
    # im_rgb_adj_pil.save(output_filename)

# Function to obtain color adjustment function
import torch
import torch.nn as nn

class MyCustomTransform(nn.Module):
    def __init__(self, RG, BY):
        super(MyCustomTransform, self).__init__()
        self.RG = RG
        self.BY = BY

    def forward(self, img):
        # img input [batch, channel, height, width]
        # device = img.device
        # image_rgb_tensor = img.unsqueeze(0).to(device)

        #################
        # RGB to CIELAB
        # first convert [batch, channel, height, width] to [height, width, channel, batch]
        image_rgb_tensor = img.permute(2, 3, 1, 0)
        image_cielab_tensor = color_utils.RGB_to_CIELAB_tensor(image_rgb_tensor)
        
        ###################
        # [height, width, channel, batch] to [batch, height, width, channels]
        image_cielab_tensor = image_cielab_tensor.permute(3, 0, 1, 2)
        

        # Subtract, centering color scales
        adjust_lab = torch.tensor([1.0, 255/2, 255/2], device=img.device)
        adj_mat = adjust_lab[None, None, None, :].expand(image_cielab_tensor.shape[0], 
                                                         image_cielab_tensor.shape[1], 
                                                         image_cielab_tensor.shape[2], 
                                                         -1)
        image_cielab_tensor = image_cielab_tensor - adj_mat

        # Scaling color from 0.0 (B/W) - 1.0 (Original) for RG BY thresholds
        adjust_lab = torch.tensor([1.0, self.RG, self.BY], device=img.device)
        adj_mat = adjust_lab[None, None, None, :].expand(image_cielab_tensor.shape[0], 
                                                         image_cielab_tensor.shape[1], 
                                                         image_cielab_tensor.shape[2], 
                                                         -1)
        image_cielab_tensor = image_cielab_tensor * adj_mat

        # Add values back in
        adjust_lab = torch.tensor([1.0, 255/2, 255/2], device=img.device)
        adj_mat = adjust_lab[None, None, None, :].expand(image_cielab_tensor.shape[0], 
                                                         image_cielab_tensor.shape[1], 
                                                         image_cielab_tensor.shape[2], 
                                                         -1)
        image_cielab_tensor = image_cielab_tensor + adj_mat

    

        # convert [batch, height, width, channel] to [height, width, channel, batch]
        image_cielab_tensor = image_cielab_tensor.permute(1, 2, 3, 0)
        image_rgb_tensor_weighted = color_utils.CIELAB_to_RGB_tensor(image_cielab_tensor)

        # [height, width, channel, batch] to [batch, channel, height, width]
        return image_rgb_tensor_weighted.permute(3, 2, 0, 1)
