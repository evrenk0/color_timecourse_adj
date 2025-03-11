import os, sys
import csv
from random import sample
import numpy as np

def run(Ecoset_folder):
    ### ENTER YOUR ECOSET PATH HERE #####
    # image_path = '/lab_data/tarrlab/common/datasets/Ecoset/'
    root_dir = '/user_data/emkonuk/Scripts/'
    image_path = os.path.join(root_dir, 'imges', 'Ecoset_filters', Ecoset_folder)
    
    train_path = os.path.join(image_path, 'train')
    val_path = os.path.join(image_path, 'val')
    test_path = os.path.join(image_path, 'test')
    
    #$image_set_name = 'Ecoset_RESIZED'
    image_set_name = Ecoset_folder
    
    #save_csv_path = '/lab_data/hendersonlab/startingblurry/csvFiles_NEW/%s'%image_set_name
    save_csv_path = os.path.join(root_dir, 'csvFiles', Ecoset_folder)
    
    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)
    
    def parse_data(datadir, target_dict = None):
        
        img_list = []
        ID_list = []
        uniqueID_list = []
    
        categ_folders = os.listdir(datadir)
        categ_folders = [f for f in categ_folders if '.DS_Store' not in f]
        assert(len(categ_folders)==565) # this is how any ecoset categs we expect to have.
    
        uniqueID_list = np.sort(categ_folders)
        
        for c in categ_folders:
    
            # within this folder, find all image files that exist here
            full_categ_folder = os.path.join(datadir, c)
            print('Looking in folder: %s'%(full_categ_folder))
            sys.stdout.flush()
            
            im_fns = os.listdir(full_categ_folder) 
            im_fns = [i for i in im_fns if '.DS_Store' not in i]
            exts = ['.JPEG','.jpeg', '.JPG', '.jpg', '.png', '.tiff']
    
            im_fns_keep = [i for i in im_fns if np.any([e in i for e in exts])]
            ignore_fns = [i for i in im_fns if not np.any([e in i for e in exts])]
            if len(ignore_fns)>0:
                print('Ignoring files:')
                print(np.array(ignore_fns))
    
            im_fns = im_fns_keep
            print('Found %d image files'%len(im_fns))
            sys.stdout.flush()
            
            # append to my growing lists here. all categories, all images
            img_list += [os.path.join(full_categ_folder, i) for i in im_fns]
            ID_list += [c for i in im_fns] # this is a list of category name for each image
            
        # construct a dictionary, where key and value correspond to ID and target
        class_n = len(uniqueID_list)
        if not target_dict:
            target_dict = dict(zip(uniqueID_list, range(class_n)))
    
        # this is a list of numerical label for each image
        label_list = [target_dict[ID_key] for ID_key in ID_list]
    
        print('\nFound %d images across %d categories\n'%(len(img_list), class_n))
        sys.stdout.flush()
            
        return img_list, label_list, class_n, target_dict
    
    # TRAIN
    # Gathering info about all training images
    img_list, label_list, class_n, target_dict = parse_data(train_path)
    total_len = len(img_list)
    lstOfInds = range(total_len)
    
    # Make CSV mapping the original labels (text names) to numerical labels
    # The names should be sorted, numbers go 0-565
    filename = os.path.join(save_csv_path, 'labelDict.csv')
    print('Writing to: %s'%filename)
    header = ["Orginal Label", "New Label"]
    all_labels = target_dict.items()
    with open(filename, 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(header)
      csvwriter.writerows(all_labels)
    
    # Make CSV of all images and labels in training set
    folder = os.path.join(save_csv_path, 'train') 
    filename = os.path.join(folder, 'imageToLabelDict.csv')
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('writing to: %s'%filename)
    header = ["Order", "Image", "Label"]
    results = []
    for i in range(total_len):
        results.append([str(i), img_list[i], label_list[i]])
    print('length of table to write is: %d'%len(results))
    print('first element is: ')
    print(results[0])
    
    with open(filename, 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(header)
      csvwriter.writerows(results)
    
    # Make a CSV for each training epoch
    # This provides a fixed set of images to load from on each epoch, so that it can be kept 
    # constant across training run.
    header = ["Epoch", "Order", "Image", "Label", "Index"]
    # We did a maximum of 300 epochs but chose random images for 1000 epochs in case
    # we wanted to play around with more epochs!
    folder = os.path.join(save_csv_path, 'train', 'imagesByEpoch') 
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    np.random.seed(645656) # want this to be reproducible later on
    # for i in range(1):
    # Set to 1 for testing
    ######################################################33
    for i in range(1000):
      inds_for_epoch = sample(lstOfInds,50000) # 50,000 images per epoch
      filename = os.path.join(folder, 'epoch%d.csv'%i)
      print('writing to: %s'%filename)
      final_result = []
      #create csv for epoch i
      for j in range(len(inds_for_epoch)):
        ind = inds_for_epoch[j]
        img = img_list[ind]
        label = label_list[ind]
        epoch = i
        order = j
        final_result.append([epoch, order, img, label, ind])
      #write into file
      with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(final_result)
    
    # Now repeating steps for validation and testing image sets.
    
    # VAL
    img_list, label_list, class_n, target_dict = parse_data(val_path, target_dict)
    total_len = len(img_list)
    
    #make csv of all images and labels
    folder = os.path.join(save_csv_path, 'val') 
    filename = os.path.join(folder, 'imageToLabelDict.csv')
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('writing to: %s'%filename)
    
    header = ["Order", "Image", "Label"]
    results = [] # make sure to zero this out first
    for i in range(total_len):
        results.append([str(i), img_list[i], label_list[i]])
    print('length of table to write is: %d'%len(results))
    print('first element is: ')
    print(results[0])
    
    with open(filename, 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(header)
      csvwriter.writerows(results)
    
    #TEST
    img_list, label_list, class_n, target_dict = parse_data(test_path, target_dict)
    total_len = len(img_list)
    
    #make csv of all images and labels
    folder = os.path.join(save_csv_path, 'test') 
    filename = os.path.join(folder, 'imageToLabelDict.csv')
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('writing to: %s'%filename)
    
    header = ["Order", "Image", "Label"]
    results = [] # make sure to zero this out first
    for i in range(total_len):
        results.append([str(i), img_list[i], label_list[i]])
    print('length of table to write is: %d'%len(results))
    print('first element is: ')
    print(results[0])
    
    with open(filename, 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(header)
      csvwriter.writerows(results)
