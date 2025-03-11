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

root_dir = '/user_data/emkonuk/Scripts'

class ImageDataset(Dataset):
    def __init__(self, img_lst, label_lst, typeEval, desired_size=224, to_bw=False, csvDir='RG05_BY05'):
        self.img_lst = img_lst
        self.label_lst = label_lst
        self.typeEval = typeEval
        self.desired_size = desired_size
        self.to_bw = to_bw
        self.csvDir = csvDir
        if self.typeEval == "train":
            self.current_indicies = range(50000)
        else:
            self.current_indicies = range(len(self.label_lst))

    def __len__(self):
        return len(self.current_indicies)

    def __getitem__(self, index):
        # print(index)
        # print(self.current_indicies[index])
        # print('Loading from to_bw = False%s'%self.img_lst[self.current_indicies[index]])
        img = Image.open(self.img_lst[self.current_indicies[index]])
        if self.to_bw:
            # if index==0:
            #     print('Converting to grayscale')
            img = img.convert('L')
        width, height = img.size
        # if index==0:
        #     print(img.getbands())
        #     print('size is: [%d x %d]'%(width, height))
        assert(width==self.desired_size)
        assert(height==self.desired_size)
        # images should already be square and at correct size
        
        # min_dim = min([width, height])
        img = torchvision.transforms.ToTensor()(img)
        # cropped_img = transforms.CenterCrop(min_dim)(img)
        # img = transforms.Resize(224)(cropped_img)
        label = self.label_lst[self.current_indicies[index]]
        return img, label

    def change_img_lst(self, epoch, csvDir):
        self.current_indicies = []
        assert(self.typeEval == 'train')
        filename = os.path.join(root_dir, 'csvFiles', csvDir, 'train', 'imagesByEpoch', \
                                'epoch' + str(epoch) + '.csv')
        open_file = open(filename)
        read_file = csv.reader(open_file, delimiter="\t")
        i = 0
        for row in read_file:
            if i == 0:
                i+=1
                continue
            parsed_row = row[0].split(',')
            self.current_indicies.append(int(parsed_row[-1]))

def load_data(typeEval, csvDir):
    filename = os.path.join(root_dir, 'csvFiles', csvDir, typeEval, 'imageToLabelDict.csv')
    
    img_lst = []
    label_lst = []
    open_file = open(filename)
    read_file = csv.reader(open_file, delimiter="\t")
    i = 0
    for row in read_file:
        if i == 0:
            i+=1
            continue
        parsed_row = row[0].split(',')
        img_lst.append(parsed_row[1])
        label_lst.append(int(parsed_row[2]))
    open_file.close()
    return img_lst, label_lst
