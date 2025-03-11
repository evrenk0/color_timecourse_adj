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
import sys

import dataset_functions
import blur_functions
import get_color_function
import resnet_model

root_dir = '/user_data/emkonuk/Scripts'



cuda = torch.cuda.is_available()
if cuda:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    num_workers = 4
    device = torch.device('cuda:0')
else:
    print('Using CPU')
    device = torch.device('cpu:0')
    num_workers = 0
    


def run(trial_number = 1, debug = False, training_type = 'ColorNoBlur', save_first_ims = True, csvDir = 'RG0_BY0'):

    debug = (debug==1)
    
    print('trial_number = %d, debug = %s, training_type = %s, csvDir = %s'%(trial_number, debug, training_type, csvDir))


    ###########################################################################
    # if 'BW' in training_type:
    #     to_bw=True;
    #     # 0 = convert ims to grayscale
    #     input_dim = 1; # 1 color channel
    # #################
    # # 3 channels for RG, BY, 
    # #################
    # else:
    #     to_bw=False;
    #     # 1 = keep in RGB form
    #     input_dim = 3; # 3 color channels
    
    ###########################################################################
    # input_dim: 
    # 1 channel for BW
    # 3 channels for RG, BY, and all else

    # to_bw converts ims to grayscale
    to_bw = False
    input_dim = 3;

    # default blur vaule should be 0
    blur = 0

    # IF BW
    if 'RG0_BY0' in csvDir:
        input_dim = 1;
        # to_bw = True
    if 'Blur' in training_type:
        blur = 1;
        # 1 = linear blur
        
    ###########################################################################

    ###########################################################################
    # if 'NoBlur' in training_type:
    #     blur = 0;
    #     # 0 = no blur at all
    # elif ('LinearBlur' in training_type) and ('NonLin' not in training_type):
    #     blur = 1;
    #     # 1 = linear timecourse of blur reduction
    # More sever blur case
    # else:
    #     blur = 2;
    #     # 2 = nonlinear timecourse of blur reduction 

   

    # Check if we do linear color adjustment
    if 'Color' in training_type:
        color_adj = 1;
    else:
        color_adj = 0
    ###########################################################################
    
    print('to_bw = %d, input_dim = %d, blur = %d, color_adj = %d'%(to_bw, input_dim, blur, color_adj))
    
    print('Loading .csv files, creating dataloaders...')
    
    # Dataset_functions needs to have the correct path to the images being loaded
    # csvfiles/ecosetRG/testtrainval/ 
    train_img_list, train_label_list = dataset_functions.load_data('train', csvDir)
    val_img_list, val_label_list = dataset_functions.load_data('val', csvDir)
    test_img_list, test_label_list = dataset_functions.load_data('test', csvDir)
    
    # changing these to look in the resized images directory
        # train_img_list = [i.split('RG')[0] + 'RG_RESIZED' + i.split('RG')[1] for i in train_img_list]
        # val_img_list = [i.split('RG')[0] + 'RG_RESIZED' + i.split('RG')[1] for i in val_img_list]
        # test_img_list = [i.split('RG')[0] + 'RG_RESIZED' + i.split('RG')[1] for i in test_img_list]
    ####################################################################################################
    # Leave folder unmodified, original strings should be reconstrcuted
    ####################################################################################################
    train_img_list = [i.split(csvDir)[0] + csvDir + i.split(csvDir)[1] for i in train_img_list]
    val_img_list = [i.split(csvDir)[0] + csvDir + i.split(csvDir)[1] for i in val_img_list]
    test_img_list = [i.split(csvDir)[0] + csvDir + i.split(csvDir)[1] for i in test_img_list]
    
    train_dataset = dataset_functions.ImageDataset(train_img_list, train_label_list, "train", to_bw = to_bw, csvDir = csvDir)
    val_dataset = dataset_functions.ImageDataset(val_img_list, val_label_list, "val", to_bw = to_bw, csvDir = csvDir)
    test_dataset = dataset_functions.ImageDataset(test_img_list, test_label_list, "test", to_bw = to_bw, csvDir = csvDir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory = True, drop_last=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory = True, drop_last=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory = True, drop_last=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)

    print('Dataset lens are:')
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    
    print('Dataloader lens are:')
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    
    print('Done creating dataloaders')

    # define the model here
    model = resnet_model.resnet50(input_dim=input_dim)
    model = nn.DataParallel(model)
    model = model.to(device)

    # set some parameters: these can be adjusted as needed
    learningRate = .1
    weightDecay = 5e-5
    # 300 to test, 1000 for final training
    numEpochs = 1000
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=.9)

    # this is my general training function
    def train(model, train_dataloader, device, debug=False, blur_func = None, color_func = None):
        num_correct_training = 0
        for step_num, (x, y) in enumerate(train_dataloader):
    
            if debug and (step_num>1):
                break

            if np.mod(step_num, 100)==0:
                print('Step %d of %d'%(step_num, len(train_dataloader)))
            
            model.train()
            optimizer.zero_grad()
    
            x, y = x.to(device), y.to(device)
            if blur_func is not None:
                x = blur_func(x)
            if color_func is not None:
                x = color_func(x)

            
            if step_num==0:
                print(x[0].shape)
                first_img = x[0].detach().cpu().numpy()
                print(first_img.shape)
                
            outputs = model(x)
    
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()
    
            model.eval()
            outputs = model(x)
            num_correct_training += (torch.argmax(outputs, axis=1) == y).sum().item()
            
        return model, num_correct_training, first_img

    train_accuracy = []
    val_accuracy = []
    # test_accuracy = []
    
    best_accuracy = 0

    
    # define where we will save the outputs of the training
    model_out_dir = os.path.join(root_dir, 'trials_new', training_type, 'Trial%d'%trial_number)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
        
    if save_first_ims:
        first_ims_dir = os.path.join(model_out_dir, 'first_ims')
        if not os.path.exists(first_ims_dir):
            os.makedirs(first_ims_dir)

    log_filename = os.path.join(model_out_dir, 'TrainingLog_%s_Trial%d.csv'%(training_type, trial_number))
    ckpt_filename = os.path.join(model_out_dir, 'Checkpoint_%s_Trial%d.pt'%(training_type, trial_number))
    
    print('Trial %d: saving results to: %s'%(trial_number, model_out_dir))
    
    
    # results = [["Epoch", "Training Acc", "Val Acc", "Test Acc", "Time", "Best Val Acc"]]
    results = [["Epoch", "Training Acc", "Val Acc", "Time", "Best Val Acc"]]
    
    
    for epoch in range(numEpochs):
    
        if debug & (epoch>1):
            break
    
        print('Training epoch %d'%(epoch))
        sys.stdout.flush()
    
        st_epoch = time.time()

        st_train = time.time()
        print('Init training')
        # training step
        model.train()
    
        print('Updating image list for this epoch')
        train_dataloader.dataset.change_img_lst(epoch, csvDir)

        if color_adj==0:
            color_func = None
        elif color_adj==1:    
            #blur_func = blur_functions.get_blur(epoch, epochs_to_final=50, sigma_nonlin=False)
            color_func = get_color_function.get_img_adj(epoch)

        print('color_func:')
        print(color_func)
        
        if blur==0:
            blur_func = None
        elif blur==1:    
            blur_func = blur_functions.get_blur(epoch, epochs_to_final=50, sigma_nonlin=False)
        elif blur==2:
            blur_func = blur_functions.get_blur(epoch, epochs_to_final=50, sigma_nonlin=True, base = 2)

        print('blur_func:')
        print(blur_func)
        
        print('Starting training function')
        sys.stdout.flush()
        start_time = time.time()
        model, num_correct_training, first_img = train(model, train_dataloader, device, debug = debug, blur_func = blur_func, color_func = color_func)

        et_train = time.time()
        print('Training took: %.5f sec'%(et_train - st_train))
        
        if np.mod(epoch, 50)==0:
            
            img_save_filename = os.path.join(first_ims_dir, 'train_epoch%d_firstim.png'%epoch)
            print('saving first image this epoch to %s'%img_save_filename)
            if first_img.shape[0]==1:
                first_img = np.tile(first_img, [3,1,1])
            # first_img = np.uint8(np.round(np.min([np.max([first_img * 255, 1]), 255])))
            # print(first_img[0,0:5, 0:5])
            # print(np.min(first_img), np.max(first_img))
            # print(first_img.dtype)
            # print(first_img.shape)
            i = np.minimum(np.maximum(np.uint8(first_img * 255), 0), 255)
            img_pil = Image.fromarray(np.moveaxis(i, [0], [2]))
            # img_pil = Image.fromarray(first_img)
            img_pil.save(img_save_filename)

        st_val = time.time()
        print('Init evaluation')
        # evaluate, on both validation and test sets
        model.eval()
        num_correct_validation = 0
        num_total_validation = 0
        num_val_batches_do = 500
        
        print('Looping over validation data set')
        sys.stdout.flush()
        for batch_num, (x, y) in enumerate(val_dataloader):
            
            if debug and (batch_num>1):
                break

            if batch_num>num_val_batches_do:
                break
                
            if np.mod(batch_num, 100)==0:
                print('Batch %d of %d'%(batch_num, len(val_dataloader)))
                sys.stdout.flush()
                
            x, y = x.to(device), y.to(device)
            if blur_func is not None:
                x = blur_func(x)

            if color_func is not None:
                x = color_func(x)
            outputs = model(x)
            num_correct_validation += (torch.argmax(outputs, axis=1) == y).sum().item()
            num_total_validation += len(x)

        et_val = time.time()
        print('Validation took: %.5f sec'%(et_val - st_val))
        
        # num_correct_test = 0
        # print('Looping over test data set')
        # sys.stdout.flush()
        # for batch_num, (x, y) in enumerate(test_dataloader):
    
        #     if debug and (batch_num>1):
        #         break

            
        #     if np.mod(batch_num, 100)==0:
        #         print('Batch %d of %d'%(batch_num, len(test_dataloader)))
        #         sys.stdout.flush()
                
        #     x, y = x.to(device), y.to(device)
        #     if blur_func is not None:
        #         x = blur_func(x)
        #     outputs = model(x)
        #     num_correct_test += (torch.argmax(outputs, axis=1) == y).sum().item()
    
        
        # compute results/performance for this epoch
        curr_train_accur = num_correct_training / len(train_dataset)
        # curr_val_accur = num_correct_validation / len(val_dataset)
        curr_val_accur = num_correct_validation / num_total_validation
        # curr_test_accur = num_correct_test / len(test_dataset)
        train_accuracy.append(curr_train_accur)
        val_accuracy.append(curr_val_accur)
        # test_accuracy.append(curr_test_accur)
        end_time = time.time()

        if (curr_val_accur > best_accuracy):
            # if this is the current best accuracy, we save the model 
            torch.save(model.state_dict(), ckpt_filename)
            best_accuracy = curr_val_accur

    
        # print('Epoch: {}, Training Accuracy: {:.2f}, Validation Accuracy: {:.2f}, Test Accuracy: {:.2f}'.format(epoch, curr_train_accur, curr_val_accur, curr_test_accur))
        print('Epoch: {}, Training Accuracy: {:.2f}, Validation Accuracy: {:.2f}'.format(epoch, curr_train_accur, curr_val_accur))
        print("Time for epoch: " + str(end_time - start_time) + "s" )
    
        results.append([epoch, curr_train_accur, curr_val_accur, str(end_time - start_time), best_accuracy])
        print('Writing results to %s'%log_filename)
        sys.stdout.flush()
        with open(log_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(results)
    
        et_epoch = time.time()
        print('Epoch took: %.5f seconds'%(et_epoch-st_epoch))


